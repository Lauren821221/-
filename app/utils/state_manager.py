import uuid
import streamlit as st

def init_state():
    defaults={"analysis":None,"material_id":None,"material_title":"학습 자료","recommendation":None,"selected_types":[],"questions":[],"grade":None,"agent_evidence":None,"upload_nonce":0,"thread_id":str(uuid.uuid4())}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def reset_main():
    for k in ["analysis","material_id","material_title","recommendation","selected_types","questions","grade","agent_evidence","learner_level"]:
        if k in st.session_state: del st.session_state[k]
    st.session_state.upload_nonce=st.session_state.get("upload_nonce",0)+1
    st.session_state.thread_id=str(uuid.uuid4()); init_state()
