"""Conversational Knowledge Base Bot - Multi-turn RAG assistant with confidence scoring & guardrails."""
  import os
  from fastapi import FastAPI
  from fastapi.responses import StreamingResponse
  from pydantic import BaseModel
  from langchain_openai import ChatOpenAI
  from langchain.memory import ConversationBufferWindowMemory
  from langchain.schema import Document
  from langchain_community.retrievers import BM25Retriever
  from dotenv import load_dotenv

  load_dotenv()
  app = FastAPI(title="Knowledge Base Bot", version="1.0.0")
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
  _sessions: dict[str, ConversationBufferWindowMemory] = {}
  CONFIDENCE_THRESHOLD = 0.4

  KNOWLEDGE_BASE = [
      Document(page_content="Our refund policy allows returns within 30 days of purchase with original receipt."),
      Document(page_content="Technical support is available Mon-Fri 9am-6pm IST at support@company.com."),
      Document(page_content="Premium plan includes unlimited API calls, priority support, and advanced analytics dashboard."),
      Document(page_content="To reset your password, visit account settings and click 'Forgot Password'."),
      Document(page_content="Billing cycles run monthly. Annual plans receive a 20% discount."),
  ]

  class ChatRequest(BaseModel):
      session_id: str
      message: str

  class ChatResponse(BaseModel):
      answer: str
      confidence: float
      used_fallback: bool

  def score_confidence(answer: str, docs: list) -> float:
      words = set(answer.lower().split())
      ctx = set(" ".join(d.page_content for d in docs).lower().split())
      return min(len(words & ctx) / max(len(words), 1), 1.0)

  def get_session(sid: str) -> ConversationBufferWindowMemory:
      if sid not in _sessions:
          _sessions[sid] = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history")
      return _sessions[sid]

  @app.post("/chat", response_model=ChatResponse)
  async def chat(req: ChatRequest):
      retriever = BM25Retriever.from_documents(KNOWLEDGE_BASE, k=3)
      docs = retriever.get_relevant_documents(req.message)
      context = "\n".join(d.page_content for d in docs)
      mem = get_session(req.session_id)
      history = mem.load_memory_variables({}).get("chat_history", [])
      prompt = f"Support assistant. Context: {context}\nHistory: {history}\nQ: {req.message}\nA:"
      answer = llm.invoke(prompt).content
      conf = score_confidence(answer, docs)
      used_fallback = conf < CONFIDENCE_THRESHOLD
      if used_fallback:
          answer = "I don't have enough information. Please contact support@company.com."
      mem.save_context({"input": req.message}, {"output": answer})
      return ChatResponse(answer=answer, confidence=round(conf, 2), used_fallback=used_fallback)

  @app.get("/health")
  def health(): return {"status": "online", "sessions": len(_sessions)}
  