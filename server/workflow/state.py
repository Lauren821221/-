from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class StudyState(TypedDict, total=False):
    task: str
    learner_level: str
    difficulty: str
    school_mode: str
    title: str
    words: list[dict]
    type_ids: list[str]
    question_count: int
    questions: list[dict]
    answers: dict

    next_agent: str
    supervisor_reason: str
    messages: Annotated[list, add_messages]
    tool_trace: list[dict]

    retrieved_context: str
    rag_matches: list[dict]
    rag_strategy: dict

    recommendation: dict
    quiz: dict
    grade: dict
    coach: dict

    review_target: str
    review: dict
    revision_count: int
