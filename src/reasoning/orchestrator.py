"""
Reasoning orchestrator — enhances local LLM output quality through
chain-of-thought, self-consistency, and reflection patterns.

Uses DSPy to programmatically optimize reasoning pipelines rather
than relying on manual prompt engineering.
"""

import dspy
from collections import Counter
from config import AppConfig


SYSTEM_PROMPT = (
    "You are a helpful, honest AI assistant. "
    "CRITICAL RULE: If you don't have reliable information about something, "
    "say 'I don't have information about that' rather than guessing or making things up. "
    "Never fabricate facts, URLs, statistics, or descriptions of things you haven't been given information about. "
    "It is always better to say you don't know than to hallucinate an answer. "
    "You cannot browse the internet or access websites. "
    "You can only answer from your training data and any context provided to you."
)


class ChainOfThought(dspy.Signature):
    """Answer a question by reasoning step-by-step. If you don't have reliable information, say so honestly rather than guessing."""
    question: str = dspy.InputField(desc="The question or task to reason about")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process. If you lack information, state that clearly.")
    answer: str = dspy.OutputField(desc="Final answer based on the reasoning. Say 'I don't have information about that' if unsure.")


class Reflection(dspy.Signature):
    """Critique and improve a previous answer."""
    question: str = dspy.InputField(desc="The original question")
    previous_answer: str = dspy.InputField(desc="The answer to critique")
    previous_reasoning: str = dspy.InputField(desc="The reasoning behind the answer")
    critique: str = dspy.OutputField(desc="What's wrong or could be improved")
    improved_reasoning: str = dspy.OutputField(desc="Improved step-by-step reasoning")
    improved_answer: str = dspy.OutputField(desc="Improved final answer")


class ReasoningOrchestrator(dspy.Module):
    """
    Orchestrates reasoning enhancement patterns on top of a base LLM.

    Techniques used:
    - Chain-of-Thought: Forces step-by-step reasoning
    - Self-Consistency: Samples multiple paths, picks majority answer
    - Reflection: Model critiques and improves its own output
    """

    def __init__(self, config: AppConfig | None = None):
        super().__init__()
        self.config = config or AppConfig()
        self.cot = dspy.ChainOfThought(ChainOfThought)
        self.reflect = dspy.ChainOfThought(Reflection)

    def reason(self, question: str) -> dict:
        """
        Full reasoning pipeline: CoT → Self-Consistency → Reflection.
        Returns dict with answer, reasoning chain, and metadata.
        """
        rc = self.config.reasoning

        # Step 1: Sample multiple reasoning paths (self-consistency)
        candidates = []
        for _ in range(rc.consistency_samples):
            result = self.cot(question=question)
            candidates.append({
                "reasoning": result.reasoning,
                "answer": result.answer,
            })

        # Step 2: Pick the most consistent answer
        answers = [c["answer"] for c in candidates]
        answer_counts = Counter(answers)
        best_answer = answer_counts.most_common(1)[0][0]
        best_candidate = next(c for c in candidates if c["answer"] == best_answer)

        # Step 3: Reflection — critique and improve (if enabled)
        if rc.enable_reflection:
            for _ in range(rc.max_reflection_rounds):
                reflection = self.reflect(
                    question=question,
                    previous_answer=best_candidate["answer"],
                    previous_reasoning=best_candidate["reasoning"],
                )
                # If the reflection produces a substantially different answer, use it
                if reflection.improved_answer.strip() != best_candidate["answer"].strip():
                    best_candidate = {
                        "reasoning": reflection.improved_reasoning,
                        "answer": reflection.improved_answer,
                        "critique": reflection.critique,
                    }

        return {
            "answer": best_candidate["answer"],
            "reasoning": best_candidate["reasoning"],
            "critique": best_candidate.get("critique"),
            "consistency_samples": len(candidates),
            "agreement_rate": answer_counts.most_common(1)[0][1] / len(candidates),
        }

    def quick_answer(self, question: str) -> dict:
        """Single-pass chain-of-thought without self-consistency or reflection."""
        result = self.cot(question=question)
        return {
            "answer": result.answer,
            "reasoning": result.reasoning,
        }


def configure_dspy(config: AppConfig | None = None):
    """Configure DSPy to use the local Ollama model."""
    config = config or AppConfig()
    lm = dspy.LM(
        model=f"ollama_chat/{config.model.reasoning_model}",
        api_base=config.model.ollama_base_url,
        temperature=config.model.reasoning_temperature,
        system_prompt=SYSTEM_PROMPT,
    )
    dspy.configure(lm=lm)
    return lm


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    console.print("\n[bold]Local LLM Reasoning Orchestrator[/bold]")
    console.print("Configuring DSPy with local Ollama model...\n")

    config = AppConfig()
    configure_dspy(config)
    orchestrator = ReasoningOrchestrator(config)

    test_questions = [
        "What is 247 * 38? Show your work.",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Write a Python function that finds the longest palindromic substring in a given string.",
    ]

    for q in test_questions:
        console.print(Panel(q, title="Question", border_style="blue"))
        result = orchestrator.reason(q)

        if config.reasoning.show_reasoning:
            console.print(Panel(
                result["reasoning"],
                title="Reasoning Chain",
                border_style="dim",
            ))
            if result.get("critique"):
                console.print(Panel(
                    result["critique"],
                    title="Self-Critique",
                    border_style="yellow",
                ))

        console.print(Panel(
            Markdown(result["answer"]),
            title=f"Answer (agreement: {result.get('agreement_rate', 1):.0%})",
            border_style="green",
        ))
        console.print()
