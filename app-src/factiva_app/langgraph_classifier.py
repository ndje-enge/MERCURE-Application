"""
LangGraph-based classification workflow for LCB-FT article analysis.

"""

from typing import TypedDict, List, Dict, Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
import json
import os
import pandas as pd


class ClassificationState(TypedDict):
    """State shared across the LangGraph workflow."""
    article_id: str
    title: str
    content: str
    source: str
    source_code: str
    publisher: str
    publication_date: str
    nom_source: str
    
    # Processing state
    chunks: List[Document]
    retriever: object
    keywords: List[str]
    
    # Classification results
    initial_classifications: Dict[str, str]  # keyword -> "oui"/"non"
    confidence_scores: Dict[str, float]  # keyword -> confidence (0-1)
    detected_keywords: List[str]
    ambiguous_cases: List[str]  # Keywords needing validation
    final_classification: Literal["oui", "non"]
    classification_reasoning: str
    
    # Metadata
    tokens_used: int
    total_cost: float
    processing_steps: List[str]


def create_lcbft_classifier_graph(
    llm: ChatOpenAI,
    embedding: HuggingFaceEmbeddings,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> StateGraph:
    """
    Create a LangGraph workflow for LCB-FT article classification.
    
    The workflow consists of:
    1. Document preparation (chunking + embedding)
    2. Initial classification (quick pass)
    3. Confidence scoring
    4. Ambiguity detection
    5. Validation (for ambiguous cases)
    6. Final decision
    """
    
    # Step 1: Prepare document
    def prepare_document(state: ClassificationState) -> ClassificationState:
        """Chunk the article and create a retriever."""
        content = state["content"]
        
        if not isinstance(content, str) or content.strip() == "":
            return {
                **state,
                "chunks": [],
                "detected_keywords": [],
                "final_classification": "non",
                "processing_steps": ["prepare_document: empty content"]
            }
        
        # Create document and chunk
        doc = [Document(page_content=content, metadata={"source": state["source"]})]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(doc)
        
        # Create retriever
        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        return {
            **state,
            "chunks": chunks,
            "retriever": retriever,
            "processing_steps": state.get("processing_steps", []) + ["prepare_document: completed"]
        }
    
    # Step 2: Initial classification
    def initial_classification(state: ClassificationState) -> ClassificationState:
        """Perform initial binary classification for each keyword."""
        retriever = state["retriever"]
        keywords = state["keywords"]
        content = state["content"]
        
        initial_classifications = {}
        tokens_used = state.get("tokens_used", 0)
        total_cost = state.get("total_cost", 0.0)
        
        # Enhanced prompt for LCB-FT context
        classifier_prompt = ChatPromptTemplate.from_template("""
        You are an expert in Anti-Money Laundering and Counter-Terrorism Financing (LCB-FT/AML-CFT).
        
        Analyze the following article and determine if it significantly discusses the topic: "{keyword}".
        
        For LCB-FT classification, consider:
        - Direct mentions of financial crimes, money laundering, or terrorism financing
        - Regulatory actions, sanctions, or compliance issues
        - Suspicious transactions or activities
        - Legal proceedings related to financial crimes
        - Regulatory changes affecting AML/CFT
        
        Context from article:
        {context}
        
        Question: Does this article significantly discuss "{keyword}" in the context of LCB-FT/AML-CFT?
        
        Respond ONLY with "Oui" (Yes) or "Non" (No). Do not provide explanations.
        """)
        
        for keyword in keywords:
            # Retrieve relevant chunks
            relevant_docs = retriever.get_relevant_documents(keyword)
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Classify
            chain = (
                classifier_prompt.partial(keyword=keyword)
                | llm
                | StrOutputParser()
            )
            
            with get_openai_callback() as cb:
                response = chain.invoke({"context": context})
                tokens_used += cb.total_tokens
                total_cost += cb.total_cost
            
            initial_classifications[keyword] = response.strip()
        
        return {
            **state,
            "initial_classifications": initial_classifications,
            "tokens_used": tokens_used,
            "total_cost": total_cost,
            "processing_steps": state.get("processing_steps", []) + ["initial_classification: completed"]
        }
    
    # Step 3: Confidence scoring
    def score_confidence(state: ClassificationState) -> ClassificationState:
        """Calculate confidence scores for each classification."""
        initial_classifications = state["initial_classifications"]
        retriever = state["retriever"]
        keywords = state["keywords"]
        
        confidence_scores = {}
        ambiguous_cases = []
        tokens_used = state.get("tokens_used", 0)
        total_cost = state.get("total_cost", 0.0)
        
        scoring_prompt = ChatPromptTemplate.from_template("""
        You are analyzing an article for LCB-FT relevance.
        
        Keyword: {keyword}
        Initial classification: {classification}
        
        Article context:
        {context}
        
        Rate your confidence in this classification on a scale of 0.0 to 1.0:
        - 0.9-1.0: Very clear and explicit mention
        - 0.7-0.9: Clear mention with some context
        - 0.5-0.7: Ambiguous or indirect mention
        - 0.3-0.5: Weak connection
        - 0.0-0.3: No relevant connection
        
        Respond with ONLY a number between 0.0 and 1.0.
        """)
        
        for keyword in keywords:
            classification = initial_classifications.get(keyword, "Non")
            
            if classification.strip().lower() in ["oui", "yes", "o"]:
                # Get context for scoring
                relevant_docs = retriever.get_relevant_documents(keyword)
                context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
                
                chain = (
                    scoring_prompt.partial(keyword=keyword, classification=classification)
                    | llm
                    | StrOutputParser()
                )
                
                with get_openai_callback() as cb:
                    score_str = chain.invoke({"context": context})
                    tokens_used += cb.total_tokens
                    total_cost += cb.total_cost
                
                try:
                    score = float(score_str.strip())
                    confidence_scores[keyword] = score
                    
                    # Mark as ambiguous if confidence is low
                    if 0.4 <= score <= 0.7:
                        ambiguous_cases.append(keyword)
                except ValueError:
                    confidence_scores[keyword] = 0.5
                    ambiguous_cases.append(keyword)
            else:
                confidence_scores[keyword] = 0.0
        
        return {
            **state,
            "confidence_scores": confidence_scores,
            "ambiguous_cases": ambiguous_cases,
            "tokens_used": tokens_used,
            "total_cost": total_cost,
            "processing_steps": state.get("processing_steps", []) + ["score_confidence: completed"]
        }
    
    # Step 4: Validate ambiguous cases
    def validate_ambiguous(state: ClassificationState) -> ClassificationState:
        """Re-validate ambiguous classifications with a more detailed analysis."""
        ambiguous_cases = state.get("ambiguous_cases", [])
        retriever = state["retriever"]
        initial_classifications = state["initial_classifications"]
        confidence_scores = state.get("confidence_scores", {})
        
        if not ambiguous_cases:
            return {
                **state,
                "processing_steps": state.get("processing_steps", []) + ["validate_ambiguous: skipped (no ambiguous cases)"]
            }
        
        tokens_used = state.get("tokens_used", 0)
        total_cost = state.get("total_cost", 0.0)
        
        validation_prompt = ChatPromptTemplate.from_template("""
        You are a senior LCB-FT analyst reviewing an ambiguous case.
        
        Keyword: {keyword}
        Initial assessment: {initial_classification}
        Confidence: {confidence}
        
        Full article content:
        {full_content}
        
        Provide a detailed analysis:
        1. Is this article relevant to "{keyword}" in the LCB-FT context?
        2. What specific evidence supports or contradicts this relevance?
        3. Final decision: "Oui" or "Non"
        
        Format your response as JSON:
        {{
            "decision": "Oui" or "Non",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """)
        
        parser = JsonOutputParser()
        
        for keyword in ambiguous_cases:
            # Get full content for validation
            full_content = state["content"][:3000]  # Limit to avoid token limits
            
            chain = (
                validation_prompt.partial(
                    keyword=keyword,
                    initial_classification=initial_classifications.get(keyword, "Non"),
                    confidence=confidence_scores.get(keyword, 0.5)
                )
                | llm
                | parser
            )
            
            with get_openai_callback() as cb:
                result = chain.invoke({"full_content": full_content})
                tokens_used += cb.total_tokens
                total_cost += cb.total_cost
            
            # Update classification and confidence
            decision = result.get("decision", "Non").strip()
            if decision.lower() in ["oui", "yes", "o"]:
                initial_classifications[keyword] = "Oui"
            else:
                initial_classifications[keyword] = "Non"
            
            confidence_scores[keyword] = float(result.get("confidence", 0.5))
        
        return {
            **state,
            "initial_classifications": initial_classifications,
            "confidence_scores": confidence_scores,
            "tokens_used": tokens_used,
            "total_cost": total_cost,
            "processing_steps": state.get("processing_steps", []) + [f"validate_ambiguous: validated {len(ambiguous_cases)} cases"]
        }
    
    # Step 5: Final decision
    def make_final_decision(state: ClassificationState) -> ClassificationState:
        """Make final classification decision based on all evidence."""
        initial_classifications = state["initial_classifications"]
        confidence_scores = state.get("confidence_scores", {})
        
        # Filter detected keywords (high confidence positives)
        detected_keywords = [
            keyword for keyword, classification in initial_classifications.items()
            if classification.strip().lower() in ["oui", "yes", "o"]
            and confidence_scores.get(keyword, 0) >= 0.6
        ]
        
        # Generate reasoning
        if detected_keywords:
            reasoning = f"Article relevant to LCB-FT: detected {len(detected_keywords)} relevant keywords ({', '.join(detected_keywords[:3])})"
            final_classification = "oui"
        else:
            reasoning = "No significant LCB-FT relevance detected"
            final_classification = "non"
        
        return {
            **state,
            "detected_keywords": detected_keywords,
            "final_classification": final_classification,
            "classification_reasoning": reasoning,
            "processing_steps": state.get("processing_steps", []) + ["make_final_decision: completed"]
        }
    
    # Build the graph
    workflow = StateGraph(ClassificationState)
    
    # Add nodes
    workflow.add_node("prepare_document", prepare_document)
    workflow.add_node("initial_classification", initial_classification)
    workflow.add_node("score_confidence", score_confidence)
    workflow.add_node("validate_ambiguous", validate_ambiguous)
    workflow.add_node("make_final_decision", make_final_decision)
    
    # Define edges
    workflow.set_entry_point("prepare_document")
    workflow.add_edge("prepare_document", "initial_classification")
    workflow.add_edge("initial_classification", "score_confidence")
    workflow.add_edge("score_confidence", "validate_ambiguous")
    workflow.add_edge("validate_ambiguous", "make_final_decision")
    workflow.add_edge("make_final_decision", END)
    
    return workflow.compile()


def process_article_with_langgraph(
    row: pd.Series,
    keywords: List[str],
    llm: ChatOpenAI,
    embedding: HuggingFaceEmbeddings
) -> Dict:
    """
    Process a single article using the LangGraph workflow.
    
    Args:
        row: Article data as pandas Series
        keywords: List of keywords to check
        llm: ChatOpenAI instance
        embedding: HuggingFaceEmbeddings instance
    
    Returns:
        Dictionary with classification results
    """
    import pandas as pd
    
    # Initialize state
    initial_state: ClassificationState = {
        "article_id": str(row.get("id", "")),
        "title": str(row.get("title", "")),
        "content": str(row.get("content", "")),
        "source": str(row.get("source", "")),
        "source_code": str(row.get("source_code", "")),
        "publisher": str(row.get("publisher", "")),
        "publication_date": str(row.get("publication_date", "")),
        "nom_source": str(row.get("nom_source", "")),
        "keywords": keywords,
        "initial_classifications": {},
        "confidence_scores": {},
        "detected_keywords": [],
        "ambiguous_cases": [],
        "final_classification": "non",
        "classification_reasoning": "",
        "tokens_used": 0,
        "total_cost": 0.0,
        "processing_steps": [],
        "chunks": [],
        "retriever": None
    }
    
    # Create and run workflow
    graph = create_lcbft_classifier_graph(llm, embedding)
    final_state = graph.invoke(initial_state)
    
    # Format output to match existing structure
    return {
        "id": final_state["article_id"],
        "title": final_state["title"],
        "source": final_state["source"],
        "source_code": final_state["source_code"],
        "publisher": final_state["publisher"],
        "content": final_state["content"],
        "publication_date": final_state["publication_date"],
        "nom_source": final_state["nom_source"],
        "expressions_detectees": final_state["detected_keywords"],
        "label": final_state["final_classification"],
        "tokens_used": final_state["tokens_used"],
        "total_cost": final_state["total_cost"],
        "confidence_scores": final_state["confidence_scores"],
        "reasoning": final_state["classification_reasoning"]
    }

