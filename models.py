from sqlalchemy import Column, Integer, String
from database import Base

# ✅ Chat Model (Stores user queries & chatbot responses)
class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    bot_response = Column(String, nullable=False)
