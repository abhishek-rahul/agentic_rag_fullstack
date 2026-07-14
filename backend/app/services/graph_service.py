from typing import List, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logger import logger
from app.domain.models import LLMAnswer
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
    generation_error: str | None
    llm_response: dict | None


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

    def retrieve_context(self, state: ChatGraphState) -> ChatGraphState:
        retrieval_result = self.rag_service.retrieve(state["message"])

        state["context"] = retrieval_result.get("context", "")
        state["sources"] = retrieval_result.get("sources", [])
        state["best_retrieval_score"] = retrieval_result.get("best_score")
        return state

    def _generate_answer(self, state: ChatGraphState) -> ChatGraphState:
        llm = self.llm_gateway.get_chat_model(state["provider"], state.get("model"))
        structured_llm = llm.with_structured_output(
            LLMAnswer,
            method="json_schema",
        )

        system_prompt = f"""
You are a helpful company assistant.
Use the retrieved company context when it is relevant.
Use the conversation memory for follow-up questions.
Keep answers clear, practical, and concise.
Set grounded_in_context to true only when the answer is directly supported by the retrieved context.

Retrieved context:
{state['context']}

Conversation memory:
{state['memory_text']}
""".strip()

        try:
            response = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=state["message"]),
                ]
            )
            if response is None:
                raise OutputParserException(
                    "Model did not return a valid structured response"
                )

            logger.info("llm_structured_response=%s", response.model_dump_json())
            state["answer"] = response.answer
            state["llm_response"] = response.model_dump()
            state["generation_error"] = None
        except (ValidationError, OutputParserException) as exc:
            state["answer"] = ""
            state["llm_response"] = None
            state["generation_error"] = str(exc)

        return state

    def _save_memory(self, state: ChatGraphState) -> ChatGraphState:
        # Save the user message here. The final assistant response is saved only
        # after output guardrails run in ChatService.
        self.memory_service.add_message(state["session_id"], "user", state["message"])
        return state

    def run(
        self,
        message: str,
        session_id: str,
        provider: str,
        model: str | None,
    ) -> dict:
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
            "generation_error": None,
            "llm_response": None,
        }
        return self.graph.invoke(
            initial_state,
            config={"recursion_limit": 10},
        )
