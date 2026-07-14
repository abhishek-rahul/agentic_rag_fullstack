from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.services.llm_gateway import LLMGateway
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService


class ChatGraphState(TypedDict):
    message: str
    session_id: str
    provider: str
    model: str | None
    memory_messages: List[dict]
    memory_text: str
    context: str
    sources: List[dict]
    answer: str
    best_retrieval_score: float | None


class GraphService:
    def __init__(self, memory_service: MemoryService, rag_service: RAGService):
        self.settings = get_settings()
        self.memory_service = memory_service
        self.rag_service = rag_service
        self.llm_gateway = LLMGateway()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("retrieve_context", self.retrieve_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("save_memory", self._save_memory)

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_answer")
        graph.add_edge("generate_answer", "save_memory")
        graph.add_edge("save_memory", END)
        return graph.compile()

    def _load_memory(self, state: ChatGraphState) -> ChatGraphState:
        memory_messages = self.memory_service.load_memory(state["session_id"])
        state["memory_messages"] = memory_messages
        state["memory_text"] = self.memory_service.format_for_prompt(memory_messages)
        return state

    def retrieve_context(self, state):
        retrieval_result = self.rag_service.retrieve(state["message"])

        state["context"] = retrieval_result.get("context", "")
        state["sources"] = retrieval_result.get("sources", [])
        state["best_retrieval_score"] = retrieval_result.get("best_score")

        return state

    def _generate_answer(self, state: ChatGraphState) -> ChatGraphState:
        llm = self.llm_gateway.get_chat_model(state["provider"], state.get("model"))

        system_prompt = f"""
You are a helpful company assistant.
Use the retrieved company context when it is relevant.
Use the conversation memory for follow-up questions.
Keep answers clear, practical, and concise.

Retrieved context:
{state['context']}

Conversation memory:
{state['memory_text']}
""".strip()
        
        #print(system_prompt)

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["message"]),
            ]
        )
        state["answer"] = response.content
        return state

    def _save_memory(self, state: ChatGraphState) -> ChatGraphState:
        self.memory_service.add_message(state["session_id"], "user", state["message"])
        self.memory_service.add_message(state["session_id"], "assistant", state["answer"])
        return state

    def run(self, message: str, session_id: str, provider: str, model: str | None) -> dict:
        initial_state: ChatGraphState = {
            "message": message,
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "memory_messages": [],
            "memory_text": "",
            "context": "",
            "sources": [],
            "answer": "",
            "best_retrieval_score": None,
        }
        return self.graph.invoke(
            initial_state,
            config={"recursion_limit": 10}
        )
