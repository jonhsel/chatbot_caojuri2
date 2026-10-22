"""
Mede o consumo de tokens do chatbot ANTES e DEPOIS do RAG e compara com os limites do modelo.

Uso:
    python avaliacao_rag/medir_tokens.py                       # tokenizador local (offline)
    python avaliacao_rag/medir_tokens.py --modo api            # contagem exata via API (exige GOOGLE_API_KEY)
    python avaliacao_rag/medir_tokens.py --modelo gemini-3.8-flash   # outro modelo de limites_modelos.json

O que é MEDIDO (a partir dos arquivos reais e do código do app):
  - tokens do acervo inteiro, de cada documento e de cada trecho (chunk);
  - tokens do prompt de sistema antes/depois do RAG;
  - tokens do contexto que o RAG realmente entrega para cada pergunta do golden_set.json
    (o mesmo contexto que o app monta: documento fixo + top-K trechos do BM25).
O que é PREMISSA (declarado no JSON, editável por parâmetro):
  - tamanho médio de pergunta/resposta (vem das conversas salvas em conversas_salvas/);
  - limites TPM/RPM/RPD do modelo na SUA chave (a documentação do Google não os publica por
    modelo: eles ficam no painel do AI Studio, por projeto).

Saída: avaliacao_rag/resultados/tokens.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime

from comum import (RAIZ, RESULTADOS, PASTA, ContadorTokens, carregar_documentos, ler_config_do_app,
                   nfc, system_message_antes_do_rag)

TURNOS_SIMULADOS = 10


def percentil(valores, p):
    valores = sorted(valores)
    if not valores:
        return 0
    i = min(len(valores) - 1, max(0, math.ceil(p / 100 * len(valores)) - 1))
    return valores[i]


def tokens_das_conversas_salvas(contar):
    """Tamanho médio (em tokens) de perguntas e respostas nas conversas salvas do app."""
    humanos, ias = [], []
    for arq in sorted((RAIZ / "conversas_salvas").glob("*.txt")):
        texto = arq.read_text(encoding="utf-8")
        for bloco in re.split(r"\n\n(?=(?:HUMAN|AI): )", texto.strip()):
            if bloco.startswith("HUMAN: "):
                humanos.append(contar(bloco[7:].strip()))
            elif bloco.startswith("AI: "):
                ias.append(contar(bloco[4:].strip()))
    return humanos, ias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="auto", choices=["auto", "local", "api", "estimativa"])
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--golden", default=str(PASTA / "golden_set.json"))
    ap.add_argument("--saida", default=str(RESULTADOS / "tokens.json"))
    ap.add_argument("--modelo", default="gemini-3.5-flash-lite",
                    help="modelo a avaliar; os limites vêm de limites_modelos.json (painel do AI Studio)")
    ap.add_argument("--tpm", type=int, default=None, help="substitui o TPM do painel (tokens de ENTRADA por minuto)")
    ap.add_argument("--rpm", type=int, default=None, help="substitui o RPM do painel")
    ap.add_argument("--rpd", type=int, default=None, help="substitui o RPD do painel")
    ap.add_argument("--janela", type=int, default=None, help="substitui o limite de tokens de entrada do modelo")
    ap.add_argument("--tokens-resposta", type=int, default=None, help="tokens médios por resposta (padrão: conversas salvas)")
    args = ap.parse_args()

    from rag import BM25Retriever

    cfg = ler_config_do_app()
    k_prod, janela_msgs = cfg["k_trechos"], cfg["janela_historico"]
    docs = carregar_documentos(cfg["arquivos"])
    presets = json.load(open(PASTA / "limites_modelos.json", encoding="utf-8"))
    preset = presets["modelos"].get(args.modelo, {})
    contar = ContadorTokens(args.modo, args.api_key, args.modelo)
    print("Contador:", contar.descricao)

    # ---------------- acervo ----------------
    por_doc = []
    for nome, conteudo in docs:
        por_doc.append({"nome": nome, "caracteres": len(conteudo), "tokens": contar(conteudo)})
    todos_juntos = "\n\n".join(c for _, c in docs)            # exatamente como a versão pré-RAG montava o prompt
    tokens_acervo = contar(todos_juntos)
    for d in por_doc:
        d["pct_do_acervo"] = d["tokens"] / tokens_acervo
    chars_por_token = len(todos_juntos) / tokens_acervo

    # ---------------- prompt de sistema ----------------
    sm_depois = cfg["system_message"]
    sm_antes = system_message_antes_do_rag(sm_depois)
    s_depois = contar(sm_depois.replace("{contexto}", ""))
    s_antes = contar(sm_antes.replace("{documentos}", ""))

    # ---------------- trechos e contexto do RAG ----------------
    retriever = BM25Retriever(docs)
    tokens_trechos = [contar(f"[Fonte: {n}]\n{t}") for n, t in retriever.trechos]
    # DOC_SEMPRE_INCLUIDO pode estar comentado no app (sem documento fixo)
    prefixo_fixo = ""
    if cfg["doc_fixo"]:
        doc_fixo_texto = next(c for n, c in docs if nfc(n) == nfc(cfg["doc_fixo"]))
        prefixo_fixo = f"[Fonte: {cfg['doc_fixo']}]\n{doc_fixo_texto}\n\n"
    tokens_doc_fixo = contar(prefixo_fixo) if prefixo_fixo else 0

    golden = json.load(open(args.golden, encoding="utf-8"))
    por_pergunta = []
    for c in golden["consultas"]:
        recuperado = retriever.contexto(c["pergunta"], k_prod)
        contexto = f"{prefixo_fixo}{recuperado}"   # igual ao app
        t_ctx, t_perg = contar(contexto), contar(c["pergunta"])
        por_pergunta.append({"id": c["id"], "categoria": c["categoria"], "pergunta": c["pergunta"],
                             "n_trechos": len(retriever.busca(c["pergunta"], k_prod)),
                             "tokens_contexto": t_ctx, "tokens_pergunta": t_perg})

    # ---------------- premissas de conversa ----------------
    humanos, ias = tokens_das_conversas_salvas(contar)
    q_med = round(statistics.mean(humanos)) if humanos else round(statistics.mean(p["tokens_pergunta"] for p in por_pergunta))
    a_med = args.tokens_resposta or (round(statistics.mean(ias)) if ias else 400)
    ctx_media = statistics.mean(p["tokens_contexto"] for p in por_pergunta)
    ctx_p95 = percentil([p["tokens_contexto"] for p in por_pergunta], 95)
    ctx_max = max(p["tokens_contexto"] for p in por_pergunta)
    ctx_min = min(p["tokens_contexto"] for p in por_pergunta)

    # ---------------- simulação de uma sessão ----------------
    pares_janela = janela_msgs // 2
    sessao = []
    for t in range(1, TURNOS_SIMULADOS + 1):
        hist_antes = (t - 1) * (q_med + a_med)                       # versão antiga: histórico inteiro
        hist_depois = min(t - 1, pares_janela) * (q_med + a_med)     # versão nova: só as últimas mensagens
        entrada_antes = s_antes + tokens_acervo + hist_antes + q_med
        entrada_depois = s_depois + ctx_media + hist_depois + q_med
        sessao.append({"turno": t,
                       "entrada_antes": round(entrada_antes), "entrada_depois": round(entrada_depois),
                       "saida": a_med})

    antes_1 = sessao[0]["entrada_antes"]
    depois_1 = sessao[0]["entrada_depois"]
    reducao = 1 - depois_1 / antes_1

    # ---------------- limites (premissas) ----------------
    tpm = args.tpm or preset.get("tpm") or 1_000_000
    rpm = args.rpm or preset.get("rpm")
    rpd = args.rpd or preset.get("rpd")
    janela = args.janela or preset.get("janela") or 1_048_576
    origem = lambda arg, chave: ("informado pelo usuário" if arg else
                                 f"painel do AI Studio ({presets['fonte']})" if preset.get(chave) else "premissa")
    limites = {
        "tpm": tpm, "rpm": rpm, "rpd": rpd, "janela_contexto": janela,
        "status": {
            "tpm": origem(args.tpm, "tpm"), "rpm": origem(args.rpm, "rpm"), "rpd": origem(args.rpd, "rpd"),
            "janela_contexto": ("informado pelo usuário" if args.janela else
                                preset["janela_fonte"] if preset.get("janela") else
                                "NÃO CONFIRMADA para este modelo; usando 1.048.576 (valor do Gemini 3.8 Flash)"),
        },
        "metrica_do_tpm": "tokens de ENTRADA por minuto (documentação de rate limits do Gemini API)",
    }
    capacidade = {
        "perguntas_por_minuto_antes": tpm // antes_1,
        "perguntas_por_minuto_depois": tpm // depois_1,
        "pct_tpm_por_pergunta_antes": antes_1 / tpm,
        "pct_tpm_por_pergunta_depois": depois_1 / tpm,
        "pct_janela_antes": antes_1 / janela,
        "pct_janela_depois": depois_1 / janela,
        # faixa de TPM compatível com a regra "2 perguntas/min" observada na versão antiga
        "tpm_implicito_faixa": [2 * antes_1, 3 * antes_1 - 1],
    }

    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": args.modelo,
        "modelo_no_app": cfg["modelo"],
        "contador": {"modo": contar.modo, "descricao": contar.descricao},
        "configuracao_do_app": {"k_trechos": k_prod, "janela_historico_msgs": janela_msgs, "doc_fixo": cfg["doc_fixo"],
                                "n_documentos": len(docs)},
        "acervo": {"tokens": tokens_acervo, "caracteres": len(todos_juntos), "chars_por_token": chars_por_token,
                   "documentos": sorted(por_doc, key=lambda d: -d["tokens"])},
        "prompt_sistema": {"antes": s_antes, "depois": s_depois},
        "doc_fixo_tokens": tokens_doc_fixo,
        "trechos": {"quantidade": len(tokens_trechos), "tokens_medio": statistics.mean(tokens_trechos),
                    "tokens_p95": percentil(tokens_trechos, 95), "tokens_max": max(tokens_trechos)},
        "premissas_conversa": {"tokens_pergunta": q_med, "tokens_resposta": a_med,
                               "fonte": f"{len(humanos)} perguntas e {len(ias)} respostas em conversas_salvas/"
                                        if not args.tokens_resposta else "tokens-resposta informado pelo usuário"},
        "contexto_rag": {"media": ctx_media, "p95": ctx_p95, "max": ctx_max, "min": ctx_min, "n_perguntas": len(por_pergunta)},
        "por_pergunta": por_pergunta,
        "sessao": sessao,
        "resumo": {"entrada_antes_turno1": antes_1, "entrada_depois_turno1": depois_1, "reducao_pct": reducao,
                   "fator": antes_1 / depois_1},
        "limites": limites,
        "capacidade": capacidade,
    }
    RESULTADOS.mkdir(exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)

    print(f"Acervo: {tokens_acervo:,} tokens ({len(todos_juntos):,} caracteres, {chars_por_token:.2f} caract./token)")
    print(f"Prompt da 1ª pergunta: ANTES {antes_1:,}  |  DEPOIS {depois_1:,}  ->  redução de {reducao:.1%} (÷{antes_1/depois_1:.0f})")
    print(f"Contexto do RAG: média {ctx_media:,.0f} | p95 {ctx_p95:,} | máx {ctx_max:,} tokens")
    print(f"Com TPM = {tpm:,}: ANTES {capacidade['perguntas_por_minuto_antes']} perg./min | DEPOIS {capacidade['perguntas_por_minuto_depois']} perg./min")
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
