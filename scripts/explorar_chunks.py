from src.utils.vector_store import DADOS, carregar_chunks

chunks = carregar_chunks(DADOS / "manual_operacoes.txt")

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content[:120], "...\n")