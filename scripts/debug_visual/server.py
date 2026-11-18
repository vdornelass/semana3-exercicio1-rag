"""Debug visual do pipeline RAG em tempo real.

Uso (na raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\debug_visual\\server.py
Depois abra http://localhost:8765
"""
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

from src.graph.builder import build_graph  # noqa: E402
from src.graph.prompt.rag_prompt import RAG_PROMPT  # noqa: E402
from src.utils.langfuse_client import flush, run_config  # noqa: E402
from src.utils.vector_store import DADOS, carregar_chunks, get_vector_store  # noqa: E402

PORTA = 8765
HTML = Path(__file__).with_name("index.html")
ORDEM_NOS = ["retrieve", "generate", "format_output"]
GRAPH = build_graph()


def _pca_2d(vetores: np.ndarray) -> np.ndarray:
    centro = vetores - vetores.mean(axis=0)
    _, _, vt = np.linalg.svd(centro, full_matrices=False)
    pts = centro @ vt[:2].T
    pts = pts / (np.abs(pts).max(axis=0) + 1e-9)
    return pts


def executar(pergunta: str, emit):
    t0 = time.perf_counter()

    def ms():
        return round((time.perf_counter() - t0) * 1000)

    emit("inicio", {"pergunta": pergunta, "t": ms()})

    # ------------------------------------------------ arquivos de dados
    arquivos = sorted(p.name for p in DADOS.glob("*.txt"))
    indexados = get_vector_store.__wrapped__.__defaults__[0]
    emit("arquivos", {
        "t": ms(),
        "arquivos": [{"nome": a, "indexado": a in indexados} for a in arquivos],
    })

    # ------------------------------------------------ chunks
    emit("etapa", {"id": "chunks", "status": "rodando", "t": ms()})
    chunks = [c for a in indexados for c in carregar_chunks(a)]
    emit("chunks", {
        "t": ms(),
        "chunks": [
            {"secao": c.metadata.get("secao", "-"), "doc": c.metadata["doc"],
             "chars": len(c.page_content), "preview": c.page_content[:160]}
            for c in chunks
        ],
    })
    emit("etapa", {"id": "chunks", "status": "ok", "t": ms()})

    # ------------------------------------------------ vector store
    emit("etapa", {"id": "index", "status": "rodando", "t": ms()})
    info = get_vector_store.cache_info()
    ti = time.perf_counter()
    store = get_vector_store()
    dur = round((time.perf_counter() - ti) * 1000, 2)
    cache_hit = get_vector_store.cache_info().hits > info.hits
    ids = list(store.store.keys())
    vetores = np.array([store.store[i]["vector"] for i in ids])
    secoes = [store.store[i]["metadata"].get("secao", "-") for i in ids]
    emit("index", {
        "t": ms(), "ms": dur, "cache_hit": cache_hit,
        "n": len(ids), "dim": int(vetores.shape[1]),
    })
    emit("etapa", {"id": "index", "status": "ok", "t": ms()})

    # ------------------------------------------------ embedding da pergunta + ranking
    emit("etapa", {"id": "embed", "status": "rodando", "t": ms()})
    ti = time.perf_counter()
    qvec = np.array(store.embedding.embed_query(pergunta))
    emit("embed", {
        "t": ms(), "ms": round((time.perf_counter() - ti) * 1000),
        "dim": len(qvec), "vetor": [round(float(v), 4) for v in qvec],
    })
    norm = np.linalg.norm(vetores, axis=1) * np.linalg.norm(qvec)
    scores = (vetores @ qvec) / norm
    ordem = np.argsort(-scores)
    pts = _pca_2d(np.vstack([vetores, qvec]))
    emit("etapa", {"id": "embed", "status": "ok", "t": ms()})

    def emitir_ranking(recuperados: set[str]):
        # "top" = chunks que o nó retrieve REALMENTE devolveu (respeita o k do seu código)
        emit("ranking", {
            "t": ms(), "k": len(recuperados),
            "itens": [{"secao": secoes[i], "score": round(float(scores[i]), 4),
                       "top": secoes[i] in recuperados} for i in ordem],
            "mapa": {
                "chunks": [{"secao": secoes[i], "x": float(pts[i][0]), "y": float(pts[i][1]),
                            "score": float(scores[i]), "top": secoes[i] in recuperados}
                           for i in range(len(ids))],
                "pergunta": {"x": float(pts[-1][0]), "y": float(pts[-1][1])},
            },
        })

    # ------------------------------------------------ grafo real (LangGraph)
    estado = {"pergunta": pergunta}
    emit("estado", {"t": ms(), "estado": estado})
    tempos = {}
    atual = ORDEM_NOS[0]
    tempos[atual] = time.perf_counter()
    emit("no", {"no": atual, "status": "rodando", "t": ms()})
    tokens_vistos = False

    for modo, dado in GRAPH.stream(estado, config=run_config(),
                                   stream_mode=["updates", "messages"]):
        if modo == "messages":
            msg, meta = dado
            texto = getattr(msg, "content", "")
            if texto and meta.get("langgraph_node") == "generate":
                tokens_vistos = True
                emit("token", {"t": ms(), "texto": texto})
            continue

        for no, saida in dado.items():
            dur = round((time.perf_counter() - tempos.get(no, time.perf_counter())) * 1000)
            resumo = {}
            for chave, valor in (saida or {}).items():
                if chave == "contextos":
                    resumo[chave] = [{"metadata": d.metadata, "page_content": d.page_content}
                                     for d in valor]
                else:
                    resumo[chave] = valor
                estado[chave] = resumo[chave]

            if no == "retrieve":
                emitir_ranking({d["metadata"].get("secao", "-") for d in resumo["contextos"]})
                ctx = "\n\n".join(d["page_content"] for d in resumo["contextos"])
                msgs = RAG_PROMPT.format_messages(contexto=ctx, pergunta=pergunta)
                emit("prompt", {"t": ms(), "mensagens": [
                    {"tipo": m.type, "conteudo": m.content} for m in msgs]})
            if no == "generate" and not tokens_vistos:
                emit("token", {"t": ms(), "texto": resumo.get("resposta", "")})

            emit("no", {"no": no, "status": "ok", "ms": dur, "t": ms(), "saida": resumo})
            emit("estado", {"t": ms(), "estado": estado})

            idx = ORDEM_NOS.index(no) if no in ORDEM_NOS else -1
            if 0 <= idx < len(ORDEM_NOS) - 1:
                atual = ORDEM_NOS[idx + 1]
                tempos[atual] = time.perf_counter()
                emit("no", {"no": atual, "status": "rodando", "t": ms()})

    flush()
    emit("fim", {"t": ms(), "resultado": estado.get("resultado", "")})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.read_bytes())
        elif url.path == "/perguntas":
            dados = (DADOS / "avaliacao.json").read_text(encoding="utf-8")
            self._send(200, "application/json; charset=utf-8", dados.encode("utf-8"))
        elif url.path == "/run":
            pergunta = parse_qs(url.query).get("q", [""])[0].strip()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(tipo, dados):
                linha = f"event: {tipo}\ndata: {json.dumps(dados, ensure_ascii=False)}\n\n"
                self.wfile.write(linha.encode("utf-8"))
                self.wfile.flush()

            try:
                executar(pergunta, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # mostra o erro na tela
                traceback.print_exc()
                try:
                    emit("erro", {"mensagem": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()})
                except Exception:
                    pass
        else:
            self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Debug visual rodando em http://localhost:{PORTA}  (Ctrl+C para parar)")
    ThreadingHTTPServer(("127.0.0.1", PORTA), Handler).serve_forever()
