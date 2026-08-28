from server.utils.config import generate_json
class StudyCoachAgent:
    def run(self,level,grade,context):
        return generate_json(f"""You are a vocabulary remediation coach.
LEARNER={level}; GRADE={grade}
RETRIEVED_RAG_CONTEXT={context}
For weak words provide meaning, synonyms, antonyms, related words, and one original age-appropriate example.
Return JSON: {{"weak_word_cards":[{{"word":"...","meaning":"...","synonyms":[],"antonyms":[],"related_words":[],"example":"..."}}],"study_tip":"..."}}""")
