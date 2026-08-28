# 안성맞춤 수준별 단어 스터디 에이전트 — 최종 과제 제출본

## 1. 문제와 차별성
영어 학습자료를 사진으로 공부할 때 단어를 다시 입력하고, 문제를 따로 만들고, 오답을 다시 정리해야 하는 반복 작업을 하나의 학습 흐름으로 자동화한다. 일반 단어 퀴즈와 달리 **사용자 실제 학습자료 이미지 → 단어 추출/수정 → 수준·시험유형별 추천 → RAG 근거 문제 생성 → Reviewer 품질검토 → 채점 → 취약단어 복습 → 학습이력 재사용**을 한 서비스에서 연결한다.

### 기존 일반 단어 퀴즈와의 차별화 기준
- **입력**: 미리 정해진 단어장 → 사용자가 실제 공부한 이미지/카메라 자료
- **개인화**: 고정 난이도 → 학습자 수준·시험유형·내신모드 기반 추천
- **품질관리**: 1회 생성 → RAG 근거 + 독립 Reviewer 검토/수정 루프
- **학습 연속성**: 일회성 퀴즈 → 취약 단어 복습 + SQLite 학습이력 재사용

## 2. Agent 역할
- **MaterialAnalyzerAgent**: 업로드/카메라 이미지에서 실제 단어·뜻·맥락을 추출한다.
- **SupervisorAgent**: 현재 task와 상태를 보고 다음 전문 Agent를 라우팅한다.
- **Retrieval Agent**: LLM `bind_tools()`로 RAG Tool 호출 필요성과 query를 결정한다.
- **RecommenderAgent**: 학습자 수준·난이도·내신모드·RAG 근거로 문제유형을 추천한다.
- **QuizGeneratorAgent**: 선택 유형과 RAG 근거를 사용해 문제를 생성한다.
- **ReviewerAgent**: 추천/문제 품질을 독립 검토하고 필요 시 1회 revision loop를 요청한다.
- **GraderAgent**: 답안을 채점하고 취약 단어를 판별한다.
- **StudyCoachAgent**: 취약 단어에 대한 맞춤 복습 정보를 생성한다.

## 3. Prompt Engineering
모든 생성 Agent는 역할(persona), 입력 슬롯(LEARNER/DIFFICULTY/SCHOOL_MODE/WORDS/RAG_CONTEXT), 제약조건, JSON 출력 스키마를 분리해 명시한다. QuizGenerator는 `FEW_SHOT` 예시를 **실제 prompt에 삽입**하고 self-check로 문제 수, 선택 유형, 4지선다, 정답 유일성, target word, usage 밑줄을 검증하도록 지시한다.

## 4. RAG
`document_processor.py`가 단어/뜻/예문/유의어/반의어/관련어와 metadata를 LangChain `Document`로 구조화한다. `vector_store.py`는 Azure OpenAI `text-embedding-3-large` 계열 embedding + FAISS로 indexing한다. `retriever.py`는 multi-query expansion, FAISS dense retrieval, BM25 sparse retrieval, learner-level metadata filter, weighted reranking, quality gate와 dense retry를 수행한다. 검색 context는 LangGraph state의 `retrieved_context`로 저장되어 추천·문제생성·Reviewer·복습 Agent에 전달된다.

## 5. LangGraph / Agentic Tool Calling
`server/workflow/graph.py`의 단일 production `StateGraph`가 Supervisor → indexing → Retrieval Agent → ToolNode → specialist → Reviewer/Revision 흐름을 실행한다. Retrieval Agent의 LLM은 `bind_tools(RAG_TOOLS)`로 `search_learning_knowledge` 호출 여부와 인자를 능동적으로 선택한다. `MemorySaver`와 UI의 `thread_id`로 멀티턴 상태를 유지한다.

## 6. UI 및 사용자 흐름
UI는 기존 서비스의 흐름을 유지한다: **사진 파일 업로드 / 카메라 촬영 / 기존 자료 → AI 자료 분석 → 단어 확인·수정 → 학습 조건 → AI 문제유형 추천 → 유형 추가/삭제 → 문제 수 → 문제 풀이 → 채점·맞춤복습 → 학습이력 조회/재사용**.

## 7. 입력 검증과 오류 처리
파일 크기/MIME, 빈 단어, 중복 단어, 문제유형 ID, 허용 문제 수, 문항 구조, 답안 dict를 API에서 검증한다. Agent/API Key 오류는 FastAPI HTTPException으로 변환해 UI에 이해 가능한 메시지로 표시한다.

## 8. 실행
과제 IDE에서는 프로젝트 루트에서 다음을 실행한다.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

`streamlit_app.py`가 로컬 FastAPI를 자동 실행하므로 별도 터미널에서 API 서버를 띄울 필요가 없다. 과제 환경의 Azure OpenAI 설정(`AOAI_ENDPOINT`, `AOAI_API_KEY`, `AOAI_DEPLOY_GPT4O`, `AOAI_DEPLOY_EMBED_3_LARGE`)을 사용한다.

## 9. 디렉터리 원칙
제출본에는 프로젝트가 한 번만 존재한다. `streamlit_app.py`는 단일 실행 진입점이고 `app/main.py`는 UI 구현, `server/`는 backend/Agent/RAG/DB로 역할이 분리되어 있다. 중첩 프로젝트 복사본과 미사용 sidebar/history 모듈은 포함하지 않는다.


## Azure OpenAI 실행 환경

본 제출본은 Gemini 의존성을 제거하고 과제 환경에서 제공되는 Azure OpenAI를 사용한다.

- LLM / Vision: `AzureChatOpenAI` + `AOAI_DEPLOY_GPT4O`
- RAG Embedding: `AzureOpenAIEmbeddings` + `AOAI_DEPLOY_EMBED_3_LARGE`
- Endpoint / 인증: `AOAI_ENDPOINT`, `AOAI_API_KEY`
- 선택 설정: `AOAI_API_VERSION` (기본값 `2024-10-21`)

API Key는 코드나 ZIP에 포함하지 않는다.


## 10. v8 실행 안정성 보완
- `streamlit_app.py`는 프로젝트 루트를 `sys.path`에 보장한 뒤 `app.main`을 정상 패키지 import하므로 IDE의 Streamlit 실행 버튼과 CLI 실행에서 동일한 import 규칙을 사용한다.
- UI 모듈은 `from app.utils.state_manager ...` 절대 import를 사용한다.
- SQLite DB는 `DATABASE_URL`이 없으면 프로젝트 로컬 `study_history.db`를 사용한다.
- `/recommend` API의 wrapper 응답을 UI가 올바르게 해제하며, 학습 이력의 복습 카드는 `coach.weak_word_cards`를 재사용한다.
- FastAPI 자동 시작 실패 시 `fastapi_backend.log`에 원인을 남긴다.


## 최종 제출 설계 보강

### End-to-End 사용자 흐름
학습자 수준 선택 → 교재 사진/카메라/기존 자료 입력 → Material Analyzer 단어 추출 → 사용자 단어 확인·수정 → Supervisor 라우팅 → RAG 검색 → Recommender 문제유형 추천 → Quiz Generator 문제 생성 → Reviewer 품질 검토/필요 시 재생성 → Grader 채점 → Study Coach 취약단어 복습 → SQLite 학습 이력 저장/재사용의 흐름으로 실제 서비스가 연결된다.

### 기술 선택 이유
- **LangGraph**: 단순 순차 호출이 아니라 Supervisor 라우팅, ToolNode, Reviewer 조건 분기, 재생성 루프, 채점 후 remediation 분기를 명시적인 상태 그래프로 관리하기 위해 사용한다.
- **Hybrid RAG**: Dense embedding 검색은 의미 유사성에 강하고 BM25는 정확한 어휘 일치에 강하므로 두 결과를 결합하고 reranking하여 문제 생성 근거의 품질을 높인다.
- **Agentic Tool Calling**: Retrieval Agent가 `bind_tools()`를 통해 필요 시 RAG Tool 호출 여부와 검색 질의를 선택하며, ToolNode가 실제 호출을 수행한다.
- **MemorySaver + SQLite**: MemorySaver는 한 학습 세션의 `thread_id` 기반 멀티턴 상태를, SQLite는 세션을 넘어서는 학습자료·성적·취약단어 이력을 담당한다.
- **Deterministic Grader**: 객관식/정답형 문제는 LLM의 임의 판단보다 정답 일치 방식이 정확하고 재현 가능하므로 deterministic grading을 적용하고, 생성형 AI는 취약점 분석과 복습 코칭에 집중한다.
- **Structured JSON Prompt**: Agent별 역할, 입력 슬롯, 출력 스키마, few-shot 예시를 분리하여 다양한 학습 조건에서도 안정적인 구조화 응답을 얻도록 설계한다.

### 기존 단어 학습 서비스와의 차별성
미리 정해진 단어장과 동일 문제를 반복 제공하는 방식이 아니라, 학습자가 **실제로 공부하는 교재 사진**을 출발점으로 단어를 추출하고, 학습자 수준·시험유형·내신 조건에 따라 문제유형과 문제를 동적으로 구성한다. 생성 문제는 Reviewer Agent가 독립적으로 검토하고, 채점 후 취약 단어를 Study Coach가 복습 카드로 연결하며 학습 이력에서 자료를 다시 사용할 수 있다.

### UI의 Agent 실행 근거
추천/문제 생성 후 `AI Agent 실행 정보` expander에서 Tool Calling 횟수, RAG 검색 근거 수, Reviewer 판정을 확인할 수 있다. 이는 학습 UX를 방해하지 않으면서 Multi-Agent/RAG/Tool Calling이 실제 서비스 경로에서 실행되었음을 보여주는 평가용 근거다.
