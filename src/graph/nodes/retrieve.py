from src.graph.state import State
from src.utils.vector_store import get_vector_store


def retrieve(state: State) -> dict:
    docs = get_vector_store().similarity_search(
        state["pergunta"],
        k=2
    )

    return {"contextos": docs}
