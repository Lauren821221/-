import json, uuid
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from server.db.database import get_db
from server.db.models import StudyHistory, StudyMaterial
from server.workflow.catalog import QUESTION_TYPES
from server.workflow.graph import study_graph
from server.workflow.agents.material_analyzer import MaterialAnalyzerAgent

router = APIRouter(prefix="/api/v1", tags=["study"])
LEVELS={"유치원","초등 저학년","초등 고학년","중학생","고등학생","성인"}
DIFFICULTIES={"쉬움","보통","어려움","TOSEL 수준","TOEFL Junior 수준","TOEFL 수준","최선어학원 유형"}
SCHOOL_MODES={"적용 안 함","중학교 내신","고등학교 내신"}
COUNTS={5,10,15,20,25,30}

def cfg(thread_id): return {"configurable":{"thread_id":thread_id or str(uuid.uuid4())}}

def clean_words(words:list[dict]) -> list[dict]:
    out=[]; seen=set()
    for item in words or []:
        if not isinstance(item,dict): continue
        word=str(item.get("word","")).strip()
        if not word or word.casefold() in seen: continue
        seen.add(word.casefold()); out.append({"word":word,"meaning":str(item.get("meaning","")).strip()})
    if not out: raise ValueError("학습 단어를 하나 이상 입력해주세요.")
    return out

class RecommendRequest(BaseModel):
    learner_level:str; difficulty:str; school_mode:str="적용 안 함"; words:list[dict]; thread_id:str|None=None
    @field_validator("learner_level")
    @classmethod
    def validate_level(cls,v):
        if v not in LEVELS: raise ValueError("지원하지 않는 학습자 수준입니다.")
        return v
    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls,v):
        if v not in DIFFICULTIES: raise ValueError("지원하지 않는 난이도/시험유형입니다.")
        return v
    @field_validator("school_mode")
    @classmethod
    def validate_school_mode(cls,v):
        if v not in SCHOOL_MODES: raise ValueError("지원하지 않는 내신 준비형입니다.")
        return v
    @field_validator("words")
    @classmethod
    def validate_words(cls,v): return clean_words(v)

class QuizRequest(RecommendRequest):
    type_ids:list[str]=Field(default_factory=list); question_count:int=10
    @field_validator("type_ids")
    @classmethod
    def validate_types(cls,v):
        bad=[x for x in v if x not in QUESTION_TYPES]
        if bad: raise ValueError(f"지원하지 않는 문제 유형: {', '.join(bad)}")
        if not v: raise ValueError("문제 유형을 하나 이상 선택해주세요.")
        return v
    @field_validator("question_count")
    @classmethod
    def validate_count(cls,v):
        if v not in COUNTS: raise ValueError("문제 수는 5/10/15/20/25/30 중에서 선택해주세요.")
        return v

class GradeRequest(BaseModel):
    learner_level:str; difficulty:str="보통"; school_mode:str="적용 안 함"; title:str="학습 자료"
    words:list[dict]; type_ids:list[str]=Field(default_factory=list); questions:list[dict]; answers:dict[str,Any]; thread_id:str|None=None
    @field_validator("learner_level")
    @classmethod
    def validate_level(cls,v):
        if v not in LEVELS: raise ValueError("지원하지 않는 학습자 수준입니다.")
        return v
    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls,v):
        if v not in DIFFICULTIES: raise ValueError("지원하지 않는 난이도/시험유형입니다.")
        return v
    @field_validator("school_mode")
    @classmethod
    def validate_school_mode(cls,v):
        if v not in SCHOOL_MODES: raise ValueError("지원하지 않는 내신 준비형입니다.")
        return v
    @field_validator("words")
    @classmethod
    def validate_words(cls,v): return clean_words(v)
    @field_validator("questions")
    @classmethod
    def validate_questions(cls,v):
        if not v: raise ValueError("채점할 문제가 없습니다.")
        for q in v:
            if not isinstance(q,dict) or "id" not in q or "answer" not in q: raise ValueError("문항 데이터 형식이 올바르지 않습니다.")
        return v
    @field_validator("answers")
    @classmethod
    def validate_answers(cls,v):
        if not isinstance(v,dict): raise ValueError("답안 형식이 올바르지 않습니다.")
        return v

def invoke(req,task):
    try:
        payload=req.model_dump(exclude={"thread_id"}); payload["task"]=task
        return study_graph.invoke(payload,config=cfg(req.thread_id))
    except HTTPException: raise
    except Exception as e:
        msg=str(e)
        if "API_KEY" in msg or "api key" in msg.lower() or "AOAI_" in msg: msg="Azure OpenAI 환경설정이 필요합니다. AOAI_ENDPOINT와 AOAI_API_KEY를 확인해주세요."
        raise HTTPException(status_code=500,detail=f"Agent 실행 중 오류가 발생했습니다: {msg}") from e

@router.get("/question-types")
def question_types(): return [{"id":k,"name":v[0],"description":v[1]} for k,v in QUESTION_TYPES.items()]

@router.post("/analyze")
async def analyze(file:UploadFile=File(...),learner_level:str=Form(...),db:Session=Depends(get_db)):
    if learner_level not in LEVELS: raise HTTPException(422,"지원하지 않는 학습자 수준입니다.")
    raw=await file.read()
    if not raw: raise HTTPException(400,"빈 파일입니다.")
    if len(raw)>15*1024*1024: raise HTTPException(413,"이미지는 15MB 이하로 올려주세요.")
    mime=file.content_type or "image/jpeg"
    if mime not in ("image/jpeg","image/png","image/webp","image/heic","image/heif"): raise HTTPException(400,"JPG, PNG, WEBP, HEIC 계열 이미지를 사용해주세요.")
    try: result=MaterialAnalyzerAgent().run(raw,mime,learner_level)
    except Exception as e: raise HTTPException(500,f"자료 분석 중 오류가 발생했습니다: {e}") from e
    words=result.get("words",[])
    if not words: raise HTTPException(422,"사진에서 영어 학습 단어를 찾지 못했습니다.")
    m=StudyMaterial(title=result.get("title") or file.filename or "학습 자료",learner_level=learner_level,summary=result.get("summary",""),words=json.dumps(words,ensure_ascii=False))
    db.add(m); db.commit(); db.refresh(m); result["material_id"]=m.id; result["title"]=m.title
    return result

@router.get("/materials")
def materials(db:Session=Depends(get_db)):
    rows=db.query(StudyMaterial).order_by(StudyMaterial.id.desc()).limit(100).all()
    return [{"id":r.id,"title":r.title,"learner_level":r.learner_level,"summary":r.summary or "","words":json.loads(r.words or "[]"),"created_at":str(r.created_at)} for r in rows]

@router.post("/recommend")
def recommend(req:RecommendRequest):
    r=invoke(req,"recommend"); return {"recommendation":r.get("recommendation",{}),"rag_matches":r.get("rag_matches",[]),"tool_trace":r.get("tool_trace",[]),"review":r.get("review",{})}

@router.post("/quiz")
def quiz(req:QuizRequest):
    r=invoke(req,"quiz"); return {"questions":r.get("questions",[]),"rag_matches":r.get("rag_matches",[]),"tool_trace":r.get("tool_trace",[]),"review":r.get("review",{})}

@router.post("/grade")
def grade(req:GradeRequest,db:Session=Depends(get_db)):
    r=invoke(req,"grade"); g=r.get("grade",{}); coach=r.get("coach",{})
    result={**g,"coach":coach,"review":r.get("review",{}),"rag_matches":r.get("rag_matches",[])}
    row=StudyHistory(title=req.title,learner_level=req.learner_level,difficulty=req.difficulty,school_mode=req.school_mode,words=json.dumps(req.words,ensure_ascii=False),quiz_types=json.dumps(req.type_ids,ensure_ascii=False),question_count=len(req.questions),score=int(g.get("score",0)),result=json.dumps(result,ensure_ascii=False))
    db.add(row); db.commit(); db.refresh(row); result["history_id"]=row.id
    return result

@router.get("/history")
def history(db:Session=Depends(get_db)):
    rows=db.query(StudyHistory).order_by(StudyHistory.id.desc()).limit(100).all(); out=[]
    for r in rows:
        result=json.loads(r.result or "{}")
        out.append({"id":r.id,"title":r.title,"learner_level":r.learner_level,"difficulty":r.difficulty,"school_mode":getattr(r,"school_mode","적용 안 함"),"quiz_types":json.loads(r.quiz_types or "[]"),"question_count":r.question_count,"score":r.score,"created_at":str(r.created_at),"words":json.loads(r.words or "[]"),"result":result})
    return out
