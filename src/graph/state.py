from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict, total=False):
    pergunta: str
    contextos: list[Document]
    resposta: str
    resultado: str