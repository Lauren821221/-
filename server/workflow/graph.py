"""Single production LangGraph for the assignment.

This is not a disconnected demo. FastAPI calls this compiled graph directly.
It demonstrates:
- LLM Supervisor routing
- LLM bind_tools() + ToolNode autonomous retrieval
- Hybrid RAG retrieval/reranking
- Independent Reviewer Agent with conditional revision
- Conditional remediation after grading
- MemorySaver + thread_id multi-turn state
"""
import json, os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from server.workflow.state import StudyState
from server.workflow.agents.supervisor import SupervisorAgent
from server.workflow.agents.recommender import RecommenderAgent
from server.workflow.agents.quiz_generator import QuizGeneratorAgent
from server.workflow.agents.grader import GraderAgent
from server.workflow.agents.study_coach import StudyCoachAgent
from server.workflow.agents.reviewer import ReviewerAgent
from server.rag.document_processor import build_documents
from server.rag.vector_store import index_documents
from server.rag.retriever import retrieve_learning_context
from server.rag.tools import search_learning_knowledge

memory = MemorySaver()
RAG_TOOLS = [search_learning_knowledge]

def _tool_llm():
    endpoint = (os.getenv("AOAI_ENDPOINT") or "").strip()
    api_key = (os.getenv("AOAI_API_KEY") or "").strip()
    if not endpoint or not api_key:
        raise RuntimeError("AOAI_ENDPOINT와 AOAI_API_KEY 환경변수가 필요합니다.")
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=(os.getenv("AOAI_DEPLOY_GPT4O") or os.getenv("AOAI_DEPLOY_GPT4O_MINI") or "gpt-4o-mini"),
        api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
        temperature=0,
    ).bind_tools(RAG_TOOLS)

def _query(state):
    words = ", ".join(x.get("word","") for x in state.get("words",[]) if x.get("word"))
    return f"Vocabulary evidence for: {words}. learner={state.get('learner_level','')} task={state.get('task','')}"

def supervisor_node(state):
    return SupervisorAgent().run(state)

def route_from_supervisor(state):
    return state.get("next_agent","recommend")

def index_node(state):
    docs = build_documents(state.get("words",[]), state.get("learner_level",""), state.get("title","user_material"))
    if docs:
        index_documents(docs)
    return {"messages":[HumanMessage(content=_query(state))]}

def rag_agent_node(state):
    system = SystemMessage(content=(
        "You are the Retrieval Agent. Decide autonomously whether the vocabulary task needs grounded evidence. "
        "When useful, call search_learning_knowledge yourself and choose its query and learner_level arguments. "
        "If a ToolMessage already exists for this request, do not call again; summarize/use that evidence."
    ))
    response = _tool_llm().invoke([system, *state.get("messages",[])])
    trace = list(state.get("tool_trace",[]))
    for call in getattr(response,"tool_calls",None) or []:
        trace.append({"tool":call.get("name"),"args":call.get("args",{}),"decided_by":"LLM bind_tools"})
    return {"messages":[response],"tool_trace":trace}

def route_after_rag_agent(state):
    last = state.get("messages",[])[-1] if state.get("messages") else None
    return "tools" if getattr(last,"tool_calls",None) else "collect"

def collect_rag_node(state):
    tool_payloads=[]
    for m in state.get("messages",[]):
        if isinstance(m, ToolMessage) or getattr(m,"type","")=="tool":
            try: tool_payloads.append(json.loads(str(m.content)))
            except Exception: pass
    if tool_payloads:
        last=tool_payloads[-1]
        return {"retrieved_context":last.get("context",""),"rag_matches":last.get("matches",[]),"rag_strategy":last.get("strategy",{})}
    # deterministic safety fallback only when the LLM elected not to call a tool
    detail=retrieve_learning_context(_query(state),state.get("learner_level",""),10,5)
    return {"retrieved_context":detail["context"],"rag_matches":detail["matches"],"rag_strategy":detail["strategy"]}

def route_to_specialist(state):
    return state.get("next_agent","recommend")

def recommend_node(state):
    result=RecommenderAgent().run(state.get("learner_level",""),state.get("difficulty",""),
        state.get("school_mode","적용 안 함"),state.get("words",[]),state.get("retrieved_context",""))
    return {"recommendation":result,"review_target":"recommend","revision_count":state.get("revision_count",0)}

def quiz_node(state):
    type_ids=state.get("type_ids",[]) or state.get("recommendation",{}).get("recommended_type_ids",[])
    result=QuizGeneratorAgent().run(state.get("learner_level",""),state.get("difficulty",""),
        state.get("school_mode","적용 안 함"),state.get("words",[]),type_ids,
        int(state.get("question_count",10)),state.get("retrieved_context",""))
    return {"quiz":result,"questions":result.get("questions",[]),"type_ids":type_ids,
            "review_target":"quiz","revision_count":state.get("revision_count",0)}

def review_node(state):
    target=state.get("review_target","quiz")
    artifact=state.get("quiz",{}) if target=="quiz" else state.get("recommendation",{})
    review=ReviewerAgent().run(target,artifact,state.get("learner_level",""),state.get("retrieved_context",""))
    return {"review":review}

def route_after_review(state):
    review=state.get("review",{})
    if review.get("decision")=="revise" and state.get("revision_count",0)<1:
        return "revise"
    return "accept"

def revision_node(state):
    # Reviewer and specialist collaborate: revision instruction becomes shared state/context.
    instruction=state.get("review",{}).get("revision_instruction","")
    context=state.get("retrieved_context","") + f"\n\nINDEPENDENT_REVIEWER_REVISION_REQUEST:\n{instruction}"
    count=state.get("revision_count",0)+1
    if state.get("review_target")=="quiz":
        type_ids=state.get("type_ids",[])
        result=QuizGeneratorAgent().run(state.get("learner_level",""),state.get("difficulty",""),
            state.get("school_mode","적용 안 함"),state.get("words",[]),type_ids,
            int(state.get("question_count",10)),context)
        return {"quiz":result,"questions":result.get("questions",[]),"revision_count":count}
    result=RecommenderAgent().run(state.get("learner_level",""),state.get("difficulty",""),
        state.get("school_mode","적용 안 함"),state.get("words",[]),context)
    return {"recommendation":result,"revision_count":count}

def grade_node(state):
    return {"grade":GraderAgent().run(state.get("questions",[]),state.get("answers",{}),state.get("words",[]))}

def needs_coach(state):
    grade=state.get("grade",{})
    weak=grade.get("weak_words") or grade.get("wrong_words") or []
    score,total=grade.get("score",0),grade.get("total",len(state.get("questions",[]))) or 1
    return "coach" if weak or (isinstance(score,(int,float)) and score<total) else "finish"

def coach_node(state):
    return {"coach":StudyCoachAgent().run(state.get("learner_level",""),state.get("grade",{}),state.get("retrieved_context",""))}

def build_graph():
    g=StateGraph(StudyState)
    for name,node in [
        ("supervisor",supervisor_node),("index",index_node),("rag_agent",rag_agent_node),
        ("rag_tools",ToolNode(RAG_TOOLS)),("collect_rag",collect_rag_node),
        ("recommend",recommend_node),("quiz",quiz_node),("review",review_node),
        ("revision",revision_node),("grade",grade_node),("coach",coach_node)]:
        g.add_node(name,node)

    g.add_edge(START,"supervisor")
    g.add_conditional_edges("supervisor",route_from_supervisor,{"recommend":"index","quiz":"index","grade":"grade"})
    g.add_edge("index","rag_agent")
    g.add_conditional_edges("rag_agent",route_after_rag_agent,{"tools":"rag_tools","collect":"collect_rag"})
    g.add_edge("rag_tools","rag_agent")
    g.add_conditional_edges("collect_rag",route_to_specialist,{"recommend":"recommend","quiz":"quiz","grade":"coach"})
    g.add_edge("recommend","review")
    g.add_edge("quiz","review")
    g.add_conditional_edges("review",route_after_review,{"revise":"revision","accept":END})
    g.add_edge("revision","review")
    g.add_conditional_edges("grade",needs_coach,{"coach":"index","finish":END})
    g.add_edge("coach",END)
    return g.compile(checkpointer=memory)

study_graph=build_graph()
