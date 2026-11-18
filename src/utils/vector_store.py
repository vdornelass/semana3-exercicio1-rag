from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.utils.llm_adapter import get_embeddings

DADOS = Path(__file__).resolve().parents[2] / "data"

DIVISOR = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "secao")],
    strip_headers=False,
)


def carregar_chunks(arquivo: str | Path) -> list[Document]:
    caminho = DADOS / arquivo

    chunks = DIVISOR.split_text(caminho.read_text(encoding="utf-8"))

    for chunk in chunks:
        chunk.metadata["doc"] = caminho.stem

    return chunks


@lru_cache
def get_vector_store(
    arquivos: tuple[str, ...] = ("manual_operacoes.txt",),
) -> InMemoryVectorStore:
    store = InMemoryVectorStore(get_embeddings())
    
    documentos = []
    for nome in arquivos:
        for doc in carregar_chunks(nome):
            documentos.append(doc)

    if documentos:
        store.add_documents(documentos)

    return store
