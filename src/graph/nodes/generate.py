from src.graph.prompt.rag_prompt import RAG_PROMPT
from src.graph.state import State
from src.utils.llm_adapter import get_llm


def generate(state: State) -> dict:
    contexto = "\n\n".join(
        doc.page_content for doc in state["contextos"]
    )

    resposta = (RAG_PROMPT | get_llm()).invoke({
        "contexto": contexto,
        "pergunta": state["pergunta"]
    })

    return {"resposta": str(resposta.content)}
