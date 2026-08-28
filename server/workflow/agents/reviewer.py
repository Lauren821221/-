from server.utils.config import generate_json

class ReviewerAgent:
    """Independent QA agent. It can accept or request one revision from a specialist."""
    def run(self, target: str, artifact: dict, learner_level: str, retrieved_context: str) -> dict:
        prompt = f"""
ROLE: You are an independent English-learning QA Reviewer Agent.
TARGET_AGENT={target}
LEARNER_LEVEL={learner_level}
ARTIFACT={artifact}
RAG_EVIDENCE={retrieved_context}

Independently decide whether the specialist output is usable.
Check: target vocabulary fidelity, learner-level fit, unambiguous answers,
four-choice rule for MCQ, evidence consistency, and pedagogical usefulness.
For usage questions, the target word in each choice must be visually marked with <u>...</u>.

Return JSON only:
{{"decision":"accept|revise","reason":"short Korean reason","revision_instruction":"specific instruction"}}
"""
        data = generate_json(prompt)
        if data.get("decision") not in {"accept", "revise"}:
            data["decision"] = "accept"
        return data
