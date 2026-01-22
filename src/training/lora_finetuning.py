"""
LoRA Fine-Tuning Module
========================
Parameter-efficient fine-tuning for Llama 3 using LoRA/QLoRA.

Based on the Aalap project specifications for Indian legal LLM training.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from loguru import logger


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    # LoRA hyperparameters
    r: int = 16                        # Rank of the update matrices
    lora_alpha: int = 32               # Scaling factor
    lora_dropout: float = 0.05         # Dropout probability
    
    # Target modules for LoRA
    target_modules: List[str] = None   # E.g., ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Quantization settings for QLoRA
    use_4bit: bool = True              # Use 4-bit quantization
    bnb_4bit_quant_type: str = "nf4"   # NF4 quantization type
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: all linear layers for comprehensive fine-tuning
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class LlamaLoRATrainer:
    """
    LoRA/QLoRA trainer for Llama 3 models.
    
    Enables fine-tuning on consumer GPUs with reduced memory footprint.
    Optimized for legal reasoning and instruction following.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        config: Optional[LoRAConfig] = None,
        output_dir: str = "./models/legal_llama"
    ):
        """
        Initialize the LoRA trainer.
        
        Args:
            model_name: Base Llama model from HuggingFace.
            config: LoRA configuration.
            output_dir: Directory for saving checkpoints.
        """
        self.model_name = model_name
        self.config = config or LoRAConfig()
        self.output_dir = output_dir
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        logger.info(f"Initialized LlamaLoRATrainer with {model_name}")
        logger.info(f"LoRA config: r={self.config.r}, alpha={self.config.lora_alpha}")
    
    def _setup_quantization(self):
        """Set up 4-bit quantization for QLoRA."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for QLoRA. "
                "Install with: pip install bitsandbytes"
            )
        
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        return bnb_config
    
    def load_model(self):
        """Load the base model with quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError(
                "transformers and peft are required. "
                "Install with: pip install transformers peft"
            )
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Setup quantization
        bnb_config = self._setup_quantization() if self.config.use_4bit else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable, total = self.peft_model.get_nb_trainable_parameters()
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def prepare_dataset(
        self,
        data_path: str,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output"
    ):
        """
        Prepare instruction dataset for training.
        
        Expected format (Aalap-style):
        {
            "instruction": "Summarize the following judgment...",
            "input": "The petitioner filed...",
            "output": "This case concerns..."
        }
        
        Args:
            data_path: Path to JSONL dataset.
            instruction_key: Key for instruction text.
            input_key: Key for input context.
            output_key: Key for expected output.
            
        Returns:
            Processed dataset.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets is required. Install with: pip install datasets")
        
        logger.info(f"Loading dataset from {data_path}")
        
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        def format_prompt(example):
            """Format into instruction prompt."""
            instruction = example.get(instruction_key, "")
            input_text = example.get(input_key, "")
            output = example.get(output_key, "")
            
            if input_text:
                prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
            
            return {"text": prompt}
        
        dataset = dataset.map(format_prompt)
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: bool = False
    ):
        """
        Run the training loop.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            resume_from_checkpoint: Whether to resume from last checkpoint.
            
        Returns:
            Training metrics.
        """
        try:
            from transformers import TrainingArguments
            from trl import SFTTrainer
        except ImportError:
            raise ImportError(
                "trl is required for SFT training. "
                "Install with: pip install trl"
            )
        
        if self.peft_model is None:
            self.load_model()
        
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if self.config.use_4bit else "adamw_torch",
            report_to="tensorboard",
            gradient_checkpointing=True,
        )
        
        trainer = SFTTrainer(
            model=self.peft_model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,
        )
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save_model()
        
        return trainer.state.log_history
    
    def save_model(self, path: Optional[str] = None):
        """Save the LoRA adapter weights."""
        save_path = path or f"{self.output_dir}/final_adapter"
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"LoRA adapter saved to {save_path}")
    
    def merge_and_save(self, output_path: str):
        """
        Merge LoRA weights with base model and save.
        
        This creates a standalone model without requiring PEFT at inference.
        """
        logger.info("Merging LoRA weights with base model...")
        
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        base_model: str = "meta-llama/Meta-Llama-3-8B"
    ):
        """
        Load a trained LoRA adapter.
        
        Args:
            adapter_path: Path to saved adapter.
            base_model: Base model name.
            
        Returns:
            LlamaLoRATrainer instance with loaded adapter.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError:
            raise ImportError("transformers and peft are required.")
        
        instance = cls(model_name=base_model)
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        instance.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter
        instance.peft_model = PeftModel.from_pretrained(
            instance.model,
            adapter_path
        )
        
        logger.info(f"Loaded LoRA adapter from {adapter_path}")
        return instance
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using the fine-tuned model.
        
        Args:
            instruction: Task instruction.
            input_text: Optional input context.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            
        Returns:
            Generated text.
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format prompt
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.peft_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        response_start = full_response.find("### Response:")
        if response_start != -1:
            return full_response[response_start + len("### Response:"):].strip()
        
        return full_response


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Fine-Tuning Module for Legal Llama")
    print("=" * 60)
    
    config = LoRAConfig()
    print(f"\nDefault LoRA Configuration:")
    print(f"  Rank (r): {config.r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target Modules: {config.target_modules}")
    print(f"  4-bit Quantization: {config.use_4bit}")
    
    print("\nUsage Example:")
    print("""
    from src.training.lora_finetuning import LlamaLoRATrainer, LoRAConfig
    
    config = LoRAConfig(r=16, lora_alpha=32, num_epochs=3)
    trainer = LlamaLoRATrainer(
        model_name="meta-llama/Meta-Llama-3-8B",
        config=config,
        output_dir="./models/legal_llama"
    )
    
    # Prepare Aalap-style instruction dataset
    dataset = trainer.prepare_dataset("data/corpora/aalap_instructions.jsonl")
    
    # Train
    trainer.load_model()
    trainer.train(dataset)
    
    # Generate response
    response = trainer.generate(
        instruction="Summarize the key legal issues in this case",
        input_text="The petitioner has filed a writ petition..."
    )
    """)
