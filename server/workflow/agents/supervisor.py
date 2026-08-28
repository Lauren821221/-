from server.utils.config import generate_json

class SupervisorAgent:
    """LLM supervisor that chooses the next specialist instead of a fixed pipeline."""

    ALLOWED = {"recommend", "quiz", "grade"}

    def run(self, state: dict) -> dict:
        requested = state.get("task", "")
        prompt = f"""You are the Supervisor Agent of a multi-agent vocabulary learning system.
Choose exactly one next specialist agent based on the user's requested task and available state.

REQUESTED_TASK={requested}
HAS_WORDS={bool(state.get('words'))}
HAS_QUESTIONS={bool(state.get('questions'))}
HAS_ANSWERS={bool(state.get('answers'))}

Agents:
- recommend: recommends assessment question types from learner profile + RAG evidence
- quiz: generates a quiz from selected types + RAG evidence
- grade: grades answers, then hands weak words to the remediation coach

Rules:
1. If answers and questions are present, prefer grade.
2. If type_ids are present and quiz creation is requested, choose quiz.
3. If recommendation is requested, choose recommend.
4. Never invent another route.
Return JSON only: {{"next_agent":"recommend|quiz|grade","reason":"short Korean reason"}}
"""
        try:
            data = generate_json(prompt)
            route = data.get("next_agent", requested)
            if route not in self.ALLOWED:
                route = requested if requested in self.ALLOWED else "recommend"
            return {"next_agent": route, "supervisor_reason": data.get("reason", "")}
        except Exception:
            # Deterministic fallback keeps the service usable if supervisor LLM routing fails.
            route = requested if requested in self.ALLOWED else "recommend"
            return {"next_agent": route, "supervisor_reason": "요청 task 기반 fallback routing"}
