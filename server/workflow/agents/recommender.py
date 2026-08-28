from server.utils.config import generate_json
from server.workflow.catalog import QUESTION_TYPES
class RecommenderAgent:
    def run(self,level,difficulty,school,words,context):
        prompt=f"""You are a diagnostic vocabulary assessment designer.
LEARNER={level}; DIFFICULTY={difficulty}; SCHOOL_MODE={school}
WORDS={words}; AVAILABLE_TYPES={QUESTION_TYPES}
RETRIEVED_RAG_CONTEXT={context}
Recommend 3-5 type IDs. Ground the recommendation in retrieved context.
Return JSON: {{"recommended_type_ids":["id"],"reason":"Korean explanation"}}"""
        return generate_json(prompt)
