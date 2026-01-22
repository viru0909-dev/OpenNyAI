"""
Semantic Chunker for Legal Documents
=====================================
Structure-aware chunking for long legal documents to fit model context limits.

Implements:
- Semantic chunking at paragraph/section boundaries
- Overlapping windows for context preservation
- Hierarchical indexing for RAG pipelines
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    parent_id: Optional[int] = None
    metadata: Optional[Dict] = None


class SemanticChunker:
    """
    Semantic chunking for legal documents.
    
    Features:
    - Structure-aware splitting at paragraph/section boundaries
    - Configurable overlap for context preservation
    - Hierarchical chunking for RAG retrieval
    """
    
    # Section header patterns for legal documents
    SECTION_PATTERNS = [
        r'^(?:\d+\.)+\s+',                    # 1. 2. 3. or 1.1 1.2
        r'^\([a-z]\)\s+',                     # (a) (b) (c)
        r'^\([ivx]+\)\s+',                    # (i) (ii) (iii)
        r'^(?:WHEREAS|THEREFORE|ORDER|JUDGMENT|FACTS|ISSUE|HELD|RATIO)\s*:?',
        r'^(?:The\s+)?(?:petitioner|respondent|appellant|defendant)',
        r'^(?:Section|Article|Order|Rule)\s+\d+',
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        split_by: str = "semantic"  # "semantic", "sentence", "character"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens/characters.
            chunk_overlap: Overlap between chunks (10-20% of chunk_size recommended).
            min_chunk_size: Minimum chunk size to avoid tiny fragments.
            split_by: Splitting strategy.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.split_by = split_by
        
        self.section_pattern = re.compile(
            '|'.join(self.SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
        
        logger.info(f"Initialized SemanticChunker: size={chunk_size}, overlap={chunk_overlap}")
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Preserve legal abbreviations
        legal_abbrevs = ['vs.', 'v.', 'no.', 'sec.', 'art.', 'hon.', 'mr.', 'ms.', 'dr.']
        
        protected = text
        for abbrev in legal_abbrevs:
            protected = protected.replace(abbrev, abbrev.replace('.', '<DOT>'))
        
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Approximate: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _find_section_breaks(self, text: str) -> List[int]:
        """Find positions of section breaks in text."""
        breaks = [0]
        
        for match in self.section_pattern.finditer(text):
            breaks.append(match.start())
        
        breaks.append(len(text))
        return sorted(set(breaks))
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of TextChunk objects.
        """
        if self.split_by == "semantic":
            return self._chunk_semantic(text)
        elif self.split_by == "sentence":
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_characters(text)
    
    def _chunk_semantic(self, text: str) -> List[TextChunk]:
        """Chunk at semantic boundaries (paragraphs/sections)."""
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_idx = 0
        
        for para in paragraphs:
            para_length = self._estimate_tokens(para)
            
            # If adding this paragraph exceeds chunk size
            if current_length + para_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                end_idx = start_idx + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_id=chunk_id
                ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-1] if len(current_chunk) > 0 else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = self._estimate_tokens(overlap_text) + para_length
                start_idx = end_idx - len(overlap_text)
                chunk_id += 1
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Chunk by sentences with overlap."""
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sent in sentences:
            sent_length = self._estimate_tokens(sent)
            
            if current_length + sent_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=0,  # Would need proper tracking
                    end_idx=len(chunk_text),
                    chunk_id=chunk_id
                ))
                
                # Keep last few sentences for overlap
                overlap_sents = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk
                current_chunk = overlap_sents + [sent]
                current_length = sum(self._estimate_tokens(s) for s in current_chunk)
                chunk_id += 1
            else:
                current_chunk.append(sent)
                current_length += sent_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_idx=0,
                end_idx=len(chunk_text),
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[TextChunk]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size * 4  # Convert tokens to chars
            
            # Try to end at a sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.min_chunk_size * 4:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
            
            start = end - (self.chunk_overlap * 4)  # Overlap in chars
        
        return chunks
    
    def create_hierarchical_chunks(
        self,
        text: str
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Create hierarchical chunks for RAG retrieval.
        
        Returns:
            Tuple of (parent_chunks, child_chunks).
            Parent chunks are larger summaries, child chunks are details.
        """
        # Create larger parent chunks
        parent_chunker = SemanticChunker(
            chunk_size=self.chunk_size * 3,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size * 2,
            split_by="semantic"
        )
        parent_chunks = parent_chunker.chunk_text(text)
        
        # Create smaller child chunks
        child_chunks = []
        for parent in parent_chunks:
            children = self.chunk_text(parent.text)
            for child in children:
                child.parent_id = parent.chunk_id
            child_chunks.extend(children)
        
        return parent_chunks, child_chunks
    
    def chunk_legal_document(
        self,
        text: str,
        preserve_sections: bool = True
    ) -> List[TextChunk]:
        """
        Specialized chunking for legal documents.
        
        Args:
            text: Legal document text.
            preserve_sections: Whether to respect section boundaries.
            
        Returns:
            List of chunks.
        """
        if not preserve_sections:
            return self.chunk_text(text)
        
        # Find section breaks
        breaks = self._find_section_breaks(text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(len(breaks) - 1):
            section_text = text[breaks[i]:breaks[i + 1]].strip()
            
            if not section_text:
                continue
            
            section_tokens = self._estimate_tokens(section_text)
            
            # If section is small enough, keep as single chunk
            if section_tokens <= self.chunk_size:
                chunks.append(TextChunk(
                    text=section_text,
                    start_idx=breaks[i],
                    end_idx=breaks[i + 1],
                    chunk_id=chunk_id,
                    metadata={"section_start": True}
                ))
                chunk_id += 1
            else:
                # Section is too large, sub-chunk it
                sub_chunks = self.chunk_text(section_text)
                for j, sub in enumerate(sub_chunks):
                    sub.chunk_id = chunk_id
                    sub.start_idx += breaks[i]
                    sub.end_idx += breaks[i]
                    sub.metadata = {"section_start": j == 0}
                    chunks.append(sub)
                    chunk_id += 1
        
        return chunks


if __name__ == "__main__":
    # Example usage
    sample_text = """
    JUDGMENT

    1. This is the first paragraph of the judgment, containing the background
    of the case and the parties involved in this matter.
    
    2. The petitioner has filed this writ petition under Article 226 of the
    Constitution of India, seeking relief against the respondent.
    
    FACTS
    
    3. The facts of the case are as follows. The petitioner is a citizen of
    India and has been aggrieved by the actions of the respondent.
    
    4. On 15th January 2024, the respondent issued an order which allegedly
    violated the fundamental rights of the petitioner.
    
    ISSUE
    
    5. The main issue for consideration is whether the impugned order passed
    by the respondent is legally sustainable.
    
    ANALYSIS
    
    6. We have heard the learned counsel for both parties and perused the
    material on record carefully.
    
    7. In the case of Kesavananda Bharati v. State of Kerala (AIR 1973 SC 1461),
    the Supreme Court laid down important principles regarding constitutional
    interpretation.
    
    ORDER
    
    8. For the reasons stated above, this writ petition is allowed.
    """
    
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_legal_document(sample_text)
    
    print("=" * 60)
    print("Semantic Chunking Demo")
    print("=" * 60)
    
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} ---")
        print(f"Length: ~{len(chunk.text)//4} tokens")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
