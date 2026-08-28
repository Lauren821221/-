from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func
from server.db.database import Base

class StudyMaterial(Base):
    __tablename__ = "study_materials"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, default="학습 자료")
    learner_level = Column(String(50), nullable=False)
    summary = Column(Text, nullable=True)
    words = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StudyHistory(Base):
    __tablename__ = "study_history"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, default="학습 자료")
    learner_level = Column(String(50), nullable=False)
    difficulty = Column(String(80), nullable=False, default="보통")
    school_mode = Column(String(80), nullable=False, default="적용 안 함")
    words = Column(Text, nullable=False, default="[]")
    quiz_types = Column(Text, nullable=False, default="[]")
    question_count = Column(Integer, default=0)
    score = Column(Integer, default=0)
    result = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
