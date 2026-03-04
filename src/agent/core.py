"""
Agent core — ties together reasoning orchestration and RAG retrieval
into a unified query pipeline.

Flow: query → classify → (retrieve if needed) → reason → verify → respond
"""

import dspy
from config import AppConfig
from src.reasoning.orchestrator import ReasoningOrchestrator, configure_dspy


class QueryClassifier(dspy.Signature):
    """Classify whether a query needs document retrieval or direct reasoning."""
    query: str = dspy.InputField(desc="The user's query")
    needs_retrieval: bool = dspy.OutputField(
        desc="True if the query would benefit from searching the knowledge base"
    )
    query_type: str = dspy.OutputField(
        desc="One of: reasoning, coding, knowledge, creative, conversation"
    )


class Agent:
    """
    The main agent that orchestrates all layers:
    1. Classifies the query type
    2. Retrieves relevant context from RAG if needed
    3. Applies reasoning enhancements
    4. Returns the final answer with metadata
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.lm = configure_dspy(self.config)
        self.orchestrator = ReasoningOrchestrator(self.config)
        self.classifier = dspy.ChainOfThought(QueryClassifier)
        self._query_engine = None
        self._cloud_manager = None
        self._using_cloud = False

    def _get_cloud_manager(self):
        """Lazily initialize cloud manager."""
        if self._cloud_manager is None and self.config.cloud.enabled:
            from src.cloud.manager import CloudManager
            self._cloud_manager = CloudManager(self.config.cloud)
        return self._cloud_manager

    def use_cloud(self, enable: bool = True):
        """Switch between local and cloud inference."""
        if enable:
            manager = self._get_cloud_manager()
            if manager and manager.is_available():
                url = manager.get_ollama_url()
                self.config.model.ollama_base_url = url
                self.lm = configure_dspy(self.config)
                self.orchestrator = ReasoningOrchestrator(self.config)
                self._using_cloud = True
                return True
            return False
        else:
            self.config.model.ollama_base_url = "http://localhost:11434"
            self.lm = configure_dspy(self.config)
            self.orchestrator = ReasoningOrchestrator(self.config)
            self._using_cloud = False
            return True

    def shutdown_cloud(self):
        """Shutdown any running cloud instance."""
        if self._cloud_manager:
            self._cloud_manager.shutdown()
            self._using_cloud = False

    def _get_query_engine(self):
        """Lazily initialize the RAG query engine."""
        if self._query_engine is None:
            try:
                from src.rag.indexer import get_query_engine
                self._query_engine = get_query_engine(self.config)
            except Exception:
                # RAG not set up yet — that's fine, skip retrieval
                self._query_engine = False
        return self._query_engine if self._query_engine is not False else None

    def query(self, user_input: str, use_reasoning: bool = True) -> dict:
        """
        Process a user query through the full pipeline.

        Args:
            user_input: The user's question or task
            use_reasoning: Whether to use full reasoning (CoT + consistency + reflection)
                          Set to False for quick answers

        Returns:
            dict with answer, reasoning, context, and metadata
        """
        result = {
            "query": user_input,
            "answer": "",
            "reasoning": None,
            "retrieved_context": None,
            "query_type": "unknown",
            "critique": None,
            "agreement_rate": None,
        }

        # Step 1: Classify the query
        try:
            classification = self.classifier(query=user_input)
            result["query_type"] = classification.query_type
            needs_retrieval = classification.needs_retrieval
        except Exception:
            needs_retrieval = False

        # Step 2: Retrieve context if needed and RAG is available
        augmented_query = user_input
        if needs_retrieval:
            query_engine = self._get_query_engine()
            if query_engine:
                try:
                    rag_response = query_engine.query(user_input)
                    context = str(rag_response)
                    if context.strip():
                        result["retrieved_context"] = context
                        augmented_query = (
                            f"Context from knowledge base:\n{context}\n\n"
                            f"Question: {user_input}\n\n"
                            f"Use the context above to inform your answer. "
                            f"If the context isn't relevant, rely on your own knowledge."
                        )
                except Exception:
                    pass

        # Step 3: Reason over the query
        if use_reasoning:
            reasoning_result = self.orchestrator.reason(augmented_query)
        else:
            reasoning_result = self.orchestrator.quick_answer(augmented_query)

        result["answer"] = reasoning_result["answer"]
        result["reasoning"] = reasoning_result.get("reasoning")
        result["critique"] = reasoning_result.get("critique")
        result["agreement_rate"] = reasoning_result.get("agreement_rate")
        result["cloud"] = self._using_cloud

        # Mark cloud as used (resets idle timer)
        if self._using_cloud and self._cloud_manager:
            self._cloud_manager.mark_used()

        return result
