"""
Legal RAG Pipeline
===================
Retrieval-Augmented Generation pipeline for legal question answering.

Combines retrieval of precedents with LLM generation for evidence-based reasoning.
"""

from typing import Dict, List, Optional

from loguru import logger


class LegalRAGPipeline:
    """
    RAG pipeline for legal applications.
    
    Components:
    - Retriever: Finds relevant precedents and statutes
    - Generator: LLM for synthesizing answers
    - Prompt Template: Structures the context for generation
    """
    
    SYSTEM_PROMPT = """You are an AI legal assistant specialized in Indian law. 
You provide accurate, well-reasoned legal analysis based on the provided context.
Always cite the specific precedents and statutory provisions you rely on.
If the context doesn't contain sufficient information, state so clearly.
Do not hallucinate or make up legal provisions."""
    
    CONTEXT_TEMPLATE = """### Relevant Precedents:
{precedents}

### Relevant Statutory Provisions:
{statutes}

### Question:
{question}

### Legal Analysis:"""
    
    def __init__(
        self,
        retriever: "LegalRetriever",
        generator: Optional[Any] = None,
        generator_api: str = "groq",  # "groq", "openai", "local"
        model_name: str = "llama3-70b-8192"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: LegalRetriever instance.
            generator: Optional custom generator.
            generator_api: API to use for generation.
            model_name: Model name for generation.
        """
        self.retriever = retriever
        self.generator = generator
        self.generator_api = generator_api
        self.model_name = model_name
        
        logger.info(f"Initialized LegalRAGPipeline with {generator_api}")
    
    def _init_generator(self):
        """Initialize the generation model."""
        if self.generator_api == "groq":
            try:
                from groq import Groq
                import os
                
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                
                self.generator = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq is required. Install with: pip install groq")
        
        elif self.generator_api == "openai":
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                
                self.generator = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
    
    def _format_precedents(self, precedents: List[Dict]) -> str:
        """Format retrieved precedents for the prompt."""
        formatted = []
        
        for i, p in enumerate(precedents, 1):
            citation = p.get("metadata", {}).get("citation", f"Document {i}")
            text = p.get("text", "")[:500]  # Truncate for context window
            formatted.append(f"{i}. [{citation}]\n{text}")
        
        return "\n\n".join(formatted) if formatted else "No relevant precedents found."
    
    def _format_statutes(self, statutes: List[Dict]) -> str:
        """Format retrieved statutes for the prompt."""
        formatted = []
        
        for i, s in enumerate(statutes, 1):
            provision = s.get("metadata", {}).get("provision", f"Provision {i}")
            text = s.get("text", "")[:300]
            formatted.append(f"{i}. {provision}\n{text}")
        
        return "\n\n".join(formatted) if formatted else "No relevant statutes found."
    
    def _generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using the LLM."""
        if self.generator is None:
            self._init_generator()
        
        if self.generator_api == "groq":
            response = self.generator.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content
        
        elif self.generator_api == "openai":
            response = self.generator.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported generator API: {self.generator_api}")
    
    def query(
        self,
        question: str,
        retrieve_precedents: int = 3,
        retrieve_statutes: int = 2,
        court_filter: Optional[str] = None
    ) -> Dict:
        """
        Answer a legal question using RAG.
        
        Args:
            question: Legal question to answer.
            retrieve_precedents: Number of precedents to retrieve.
            retrieve_statutes: Number of statutes to retrieve.
            court_filter: Optional court filter.
            
        Returns:
            Dictionary with answer, sources, and metadata.
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Retrieve relevant documents
        precedents = self.retriever.retrieve_precedents(
            case_facts=question,
            k=retrieve_precedents
        )
        
        statutes = self.retriever.retrieve_statutes(
            legal_issue=question,
            k=retrieve_statutes
        )
        
        # Format context
        precedents_text = self._format_precedents(precedents)
        statutes_text = self._format_statutes(statutes)
        
        prompt = self.CONTEXT_TEMPLATE.format(
            precedents=precedents_text,
            statutes=statutes_text,
            question=question
        )
        
        # Generate answer
        answer = self._generate(prompt, self.SYSTEM_PROMPT)
        
        return {
            "question": question,
            "answer": answer,
            "sources": {
                "precedents": [
                    {
                        "citation": p.get("metadata", {}).get("citation", "Unknown"),
                        "text_preview": p.get("text", "")[:200]
                    }
                    for p in precedents
                ],
                "statutes": [
                    {
                        "provision": s.get("metadata", {}).get("provision", "Unknown"),
                        "text_preview": s.get("text", "")[:200]
                    }
                    for s in statutes
                ]
            },
            "metadata": {
                "num_precedents": len(precedents),
                "num_statutes": len(statutes),
                "model": self.model_name
            }
        }
    
    def summarize_case(
        self,
        case_text: str,
        max_length: int = 500
    ) -> str:
        """
        Summarize a legal case with context from similar cases.
        
        Args:
            case_text: Full case text.
            max_length: Maximum summary length.
            
        Returns:
            Case summary.
        """
        # Find similar cases for context
        similar = self.retriever.retrieve(case_text[:1000], k=2)
        
        prompt = f"""Summarize the following legal case. 
Focus on: Issue, Key Facts, Ratio Decidendi, and Final Order.

CASE TEXT:
{case_text[:3000]}

SUMMARY:"""
        
        return self._generate(prompt, self.SYSTEM_PROMPT)
    
    def draft_argument(
        self,
        case_facts: str,
        party: str = "petitioner",
        legal_provisions: Optional[List[str]] = None
    ) -> str:
        """
        Draft legal arguments for a party.
        
        Args:
            case_facts: Facts of the case.
            party: "petitioner" or "respondent".
            legal_provisions: Specific provisions to rely on.
            
        Returns:
            Drafted arguments.
        """
        # Retrieve relevant precedents
        precedents = self.retriever.retrieve_precedents(case_facts, k=3)
        precedents_text = self._format_precedents(precedents)
        
        provisions_text = ", ".join(legal_provisions) if legal_provisions else "relevant provisions"
        
        prompt = f"""Draft legal arguments for the {party} based on the following:

CASE FACTS:
{case_facts}

RELEVANT PRECEDENTS:
{precedents_text}

PROVISIONS TO RELY ON:
{provisions_text}

ARGUMENTS FOR {party.upper()}:"""
        
        return self._generate(prompt, self.SYSTEM_PROMPT)


if __name__ == "__main__":
    print("=" * 60)
    print("Legal RAG Pipeline")
    print("=" * 60)
    print("\nUsage Example:")
    print("""
    from src.rag import VectorStore, LegalRetriever, LegalRAGPipeline
    
    # Setup
    vectorstore = VectorStore(backend="chroma")
    vectorstore.connect()
    
    retriever = LegalRetriever(vectorstore)
    rag = LegalRAGPipeline(retriever, generator_api="groq")
    
    # Query
    result = rag.query(
        "What are the legal grounds for challenging wrongful termination?"
    )
    
    print(result["answer"])
    print("\\nSources:", result["sources"])
    """)
