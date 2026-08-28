from server.utils.config import generate_json
from server.workflow.catalog import QUESTION_TYPES
FEW_SHOT = """
FEW-SHOT EXEMPLARS (format/quality examples only; never copy them):
1) meaning: target=maintain
Q: Which meaning is closest to "maintain"?
Choices: ["유지하다","발견하다","거절하다","나누다"] Answer: "유지하다"

2) usage: target=responsible
Q: Which sentence uses the underlined word awkwardly?
Choices:
["She is <u>responsible</u> for the report.",
 "The manager is <u>responsible</u> for safety.",
 "He ate a <u>responsible</u> sandwich for lunch.",
 "Parents are <u>responsible</u> for their children."]
Answer: "He ate a <u>responsible</u> sandwich for lunch."

SELF-CHECK before returning JSON:
- exactly requested count
- only selected type_ids
- every MCQ has exactly 4 choices
- one unambiguous answer
- target word comes from STUDIED_WORDS
- usage choices visibly mark target word with <u>...</u>
- learner level and RAG evidence are respected
"""

class QuizGeneratorAgent:
    def run(self,level,difficulty,school,words,type_ids,count,context):
        selected=[(x,QUESTION_TYPES[x][0]) for x in type_ids if x in QUESTION_TYPES]
        prompt=f"""You are an expert vocabulary quiz-generation agent.
LEARNER={level}; DIFFICULTY={difficulty}; SCHOOL_MODE={school}
STUDIED_WORDS={words}; SELECTED_TYPES={selected}; COUNT={count}
RETRIEVED_RAG_CONTEXT:
{context}
RULES:
1 Generate exactly {count} original questions; studied words are primary targets.
2 Use only selected type IDs.
3 MC has exactly four plausible choices; short_answer is written response.
4 For usage questions, mark every judged target occurrence as <u>target word</u>.
5 Use RAG context for meaning/examples/synonyms/antonyms/usage and do not contradict it.
6 Adapt language to learner level and difficulty; school modes should resemble Korean school assessment skills.
7 Do not copy official copyrighted questions.
FEW-SHOT QUALITY EXEMPLARS:
{FEW_SHOT}
SELF-CHECK: exact count; selected types only; 4 MC choices; answer in choices; usage underline; studied target; appropriate level.
Return JSON only: {{"questions":[{{"id":1,"type_id":"...","type_name":"...","format":"multiple_choice","question":"...","choices":["A","B","C","D"],"answer":"...","target_word":"...","explanation":"..."}}]}}"""
        return {"questions":generate_json(prompt).get("questions",[])[:count]}
