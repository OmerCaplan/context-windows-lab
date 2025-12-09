"""
Document Generator for Context Window Experiments.

Creates synthetic documents with embedded facts for testing
LLM retrieval and context window behavior.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FactPosition(Enum):
    """Position where the critical fact is embedded in the document."""
    START = "start"
    MIDDLE = "middle"
    END = "end"


@dataclass
class GeneratedDocument:
    """
    A synthetic document with an embedded critical fact.
    
    Attributes:
        content: The full document text
        fact: The critical fact embedded in the document
        fact_position: Where the fact was placed (start/middle/end)
        word_count: Number of words in the document
        document_id: Unique identifier for this document
    """
    content: str
    fact: str
    fact_position: FactPosition
    word_count: int
    document_id: int


class DocumentGenerator:
    """
    Generates synthetic documents with embedded facts for experiments.
    
    Creates realistic filler text and embeds critical facts at specified
    positions to test "Lost in the Middle" and context window phenomena.
    """
    
    # Filler text templates for generating realistic documents
    FILLER_TEMPLATES = [
        "The company reported strong quarterly results with revenue growth exceeding expectations.",
        "Research and development investments continued to drive innovation across product lines.",
        "Market analysts noted positive trends in consumer engagement metrics.",
        "Strategic partnerships announced during the period strengthened market position.",
        "Operational efficiency improvements contributed to margin expansion.",
        "Digital transformation initiatives progressed according to planned timelines.",
        "Customer satisfaction scores reached new highs across all segments.",
        "Supply chain optimization efforts yielded significant cost reductions.",
        "New product launches received positive reception in target markets.",
        "Sustainability initiatives advanced toward carbon neutrality goals.",
        "Employee engagement programs showed measurable positive outcomes.",
        "Technology infrastructure upgrades enhanced system reliability.",
        "Geographic expansion plans proceeded with new market entries.",
        "Regulatory compliance remained strong across all operating regions.",
        "Brand awareness campaigns generated increased market visibility.",
        "Quality assurance processes maintained industry-leading standards.",
        "Innovation pipeline showed promising developments for future growth.",
        "Financial position remained robust with strong liquidity ratios.",
        "Competitive positioning improved through differentiated offerings.",
        "Long-term strategic vision guided resource allocation decisions.",
    ]
    
    # Critical facts for embedding
    CRITICAL_FACTS = [
        ("The CEO of the company is David Cohen.", "David Cohen"),
        ("The company headquarters is located in Tel Aviv.", "Tel Aviv"),
        ("The annual revenue reached 50 million dollars.", "50 million dollars"),
        ("The product launch date is scheduled for March 15th.", "March 15th"),
        ("The company was founded in 1998.", "1998"),
        ("The chief scientist is Dr. Sarah Miller.", "Dr. Sarah Miller"),
        ("The research budget is 12 million euros.", "12 million euros"),
        ("The main competitor is TechCorp Industries.", "TechCorp Industries"),
        ("The patent portfolio contains 147 patents.", "147 patents"),
        ("The customer base includes 2.5 million users.", "2.5 million users"),
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the document generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
    
    def generate_filler_text(self, target_words: int) -> str:
        """
        Generate filler text with approximately the target word count.
        
        Args:
            target_words: Approximate number of words to generate
            
        Returns:
            Filler text string
        """
        sentences = []
        current_words = 0
        
        while current_words < target_words:
            sentence = self.rng.choice(self.FILLER_TEMPLATES)
            sentences.append(sentence)
            current_words += len(sentence.split())
        
        return " ".join(sentences)
    
    def embed_fact(
        self, 
        text: str, 
        fact: str, 
        position: FactPosition
    ) -> str:
        """
        Embed a fact at the specified position in the text.
        
        Args:
            text: The base text to embed the fact in
            fact: The critical fact to embed
            position: Where to place the fact (start/middle/end)
            
        Returns:
            Text with the embedded fact
        """
        sentences = text.split(". ")
        
        if position == FactPosition.START:
            return f"{fact} {text}"
        elif position == FactPosition.END:
            return f"{text} {fact}"
        else:  # MIDDLE
            mid_point = len(sentences) // 2
            sentences.insert(mid_point, fact)
            return ". ".join(sentences)
    
    def generate_document(
        self,
        document_id: int,
        words_per_doc: int = 200,
        fact_position: Optional[FactPosition] = None,
        fact_index: Optional[int] = None,
    ) -> GeneratedDocument:
        """
        Generate a single document with an embedded fact.
        
        Args:
            document_id: Unique ID for this document
            words_per_doc: Target word count for the document
            fact_position: Where to place the fact (random if None)
            fact_index: Which fact to use (random if None)
            
        Returns:
            GeneratedDocument with embedded fact
        """
        # Select fact
        if fact_index is None:
            fact_index = self.rng.randint(0, len(self.CRITICAL_FACTS) - 1)
        fact_text, expected_answer = self.CRITICAL_FACTS[fact_index]
        
        # Select position
        if fact_position is None:
            fact_position = self.rng.choice(list(FactPosition))
        
        # Generate filler and embed fact
        filler = self.generate_filler_text(words_per_doc - len(fact_text.split()))
        content = self.embed_fact(filler, fact_text, fact_position)
        
        return GeneratedDocument(
            content=content,
            fact=fact_text,
            fact_position=fact_position,
            word_count=len(content.split()),
            document_id=document_id,
        )
    
    def generate_document_set(
        self,
        num_documents: int = 5,
        words_per_doc: int = 200,
        position_distribution: Optional[dict[FactPosition, int]] = None,
    ) -> list[GeneratedDocument]:
        """
        Generate a set of documents with controlled fact positions.
        
        Args:
            num_documents: Number of documents to generate
            words_per_doc: Target word count per document
            position_distribution: Optional dict specifying how many docs
                                  should have facts at each position
                                  
        Returns:
            List of GeneratedDocuments
        """
        documents = []
        
        if position_distribution:
            # Use specified distribution
            positions = []
            for pos, count in position_distribution.items():
                positions.extend([pos] * count)
            self.rng.shuffle(positions)
        else:
            # Random positions
            positions = [self.rng.choice(list(FactPosition)) for _ in range(num_documents)]
        
        for i, position in enumerate(positions):
            doc = self.generate_document(
                document_id=i,
                words_per_doc=words_per_doc,
                fact_position=position,
                fact_index=i % len(self.CRITICAL_FACTS),
            )
            documents.append(doc)
        
        return documents
    
    def create_needle_haystack_context(
        self,
        num_documents: int = 5,
        needle_position: FactPosition = FactPosition.MIDDLE,
        words_per_doc: int = 200,
    ) -> tuple[str, str, str]:
        """
        Create a "needle in haystack" context for testing.
        
        Args:
            num_documents: Total number of documents (haystack size)
            needle_position: Where the needle (critical fact) should be
            words_per_doc: Words per document
            
        Returns:
            Tuple of (full_context, query, expected_answer)
        """
        documents = []
        needle_doc_index = {
            FactPosition.START: 0,
            FactPosition.MIDDLE: num_documents // 2,
            FactPosition.END: num_documents - 1,
        }[needle_position]
        
        fact_text, expected_answer = self.CRITICAL_FACTS[0]
        
        for i in range(num_documents):
            if i == needle_doc_index:
                # This document contains the needle
                filler = self.generate_filler_text(words_per_doc - len(fact_text.split()))
                content = self.embed_fact(filler, fact_text, FactPosition.MIDDLE)
            else:
                # Just filler
                content = self.generate_filler_text(words_per_doc)
            
            documents.append(f"Document {i + 1}:\n{content}")
        
        full_context = "\n\n".join(documents)
        query = "Who is the CEO of the company? Answer with just the name."
        
        return full_context, query, expected_answer
