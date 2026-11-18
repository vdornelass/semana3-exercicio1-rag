from src.graph.state import State


def format_output(state: State) -> dict:
    fontes = ", ".join(
        f"{doc.metadata['doc']}#{doc.metadata.get('secao', '-')}"
        for doc in state["contextos"]
    )

    return {
        "resultado": (
            f"Resposta: {state['resposta']}\n"
            f"Fontes: {fontes}"
        )
    }
