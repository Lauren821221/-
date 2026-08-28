class GraderAgent:
    def run(self,questions,answers,words):
        correct=0; wrong=[]
        for q in questions:
            actual=str(answers.get(str(q.get("id")),"")).strip(); expected=str(q.get("answer","")).strip()
            ok=actual.casefold()==expected.casefold(); correct+=int(ok)
            if not ok:wrong.append({"question_id":q.get("id"),"target_word":q.get("target_word",""),"your_answer":actual,"answer":expected,"explanation":q.get("explanation","")})
        weak=list(dict.fromkeys(x["target_word"] for x in wrong if x["target_word"]))
        return {"score":correct,"total":len(questions),"weak_words":weak,"wrong":wrong}
