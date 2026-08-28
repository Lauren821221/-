import json
from pathlib import Path
from langchain_core.documents import Document
def doc(x,source,level=""):
    text=f"""word: {x.get('word','')}
meaning: {x.get('meaning','')}
example: {x.get('example','')}
synonyms: {', '.join(x.get('synonyms',[]) or [])}
antonyms: {', '.join(x.get('antonyms',[]) or [])}
related: {', '.join(x.get('related',[]) or [])}"""
    return Document(page_content=text,metadata={"word":x.get("word",""),"level":x.get("level") or level,"source":source,"content_type":"vocabulary"})
def build_documents(words,level,source="user_material"):
    return [doc(x,source,level) for x in words if x.get("word")]
def load_seed_documents():
    p=Path(__file__).resolve().parents[2]/"knowledge"/"vocabulary_learning.json"
    return [doc(x,"seed_knowledge",x.get("level","")) for x in json.loads(p.read_text(encoding="utf-8"))]
