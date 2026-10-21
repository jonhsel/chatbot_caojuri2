"""
Avalia o recuperador BM25 (rag.py) com Precision@k, Recall@k, MRR, MAP e NDCG.

Uso:
    python avaliacao_rag/avaliar_retrieval.py
    python avaliacao_rag/avaliar_retrieval.py --golden outro_golden.json

Para cada pergunta do golden_set.json:
  1. roda o MESMO BM25Retriever que o app usa (indexando os mesmos 16 documentos);
  2. classifica cada trecho do acervo em grau 2 (tem a resposta), 1 (contexto útil) ou 0;
  3. calcula as métricas sobre o ranking devolvido para k = 1..10.
Saída: avaliacao_rag/resultados/retrieval.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

import comum
from comum import RESULTADOS, PASTA, ler_config_do_app, carregar_documentos, nfc, normaliza
from metricas_recuperacao import agregar, avaliar_consulta, media

KS = list(range(1, 11))
K_MAX = max(KS)


def compila(entradas):
    """[(arquivo_nfc, regex_compilada_sobre_texto_normalizado)]"""
    saida = []
    for e in entradas:
        padrao = e["padrao"]
        if re.search(r"\\[SWDB]", padrao):
            raise ValueError(f"Use apenas escapes minúsculos (\\s, \\w, \\d) no padrão: {padrao}")
        saida.append((nfc(e["arquivo"]), re.compile(normaliza(padrao))))
    return saida


def graus_do_acervo(trechos_norm, resposta, apoio):
    """grau (0/1/2) de cada trecho do acervo para uma pergunta + quantos trechos casam por entrada."""
    contagem = [0] * len(resposta)
    graus = []
    for nome, texto in trechos_norm:
        grau = 0
        for j, (arq, rx) in enumerate(resposta):
            if nome == arq and rx.search(texto):
                grau = 2
                contagem[j] += 1
        if grau == 0:
            for arq, rx in apoio:
                if nome == arq and rx.search(texto):
                    grau = 1
                    break
        graus.append(grau)
    return graus, contagem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(PASTA / "golden_set.json"))
    ap.add_argument("--saida", default=str(RESULTADOS / "retrieval.json"))
    args = ap.parse_args()

    from rag import BM25Retriever  # o mesmo recuperador do app

    cfg = ler_config_do_app()
    k_prod = cfg["k_trechos"]
    ks = sorted(set(KS) | {k_prod})

    docs = carregar_documentos(cfg["arquivos"])
    retriever = BM25Retriever(docs)
    trechos_norm = [(nfc(n), normaliza(t)) for n, t in retriever.trechos]
    idx_por_objeto = {id(t): i for i, t in enumerate(retriever.trechos)}
    idx_doc_fixo = {i for i, (n, _) in enumerate(trechos_norm) if cfg["doc_fixo"] and n == nfc(cfg["doc_fixo"])}

    golden = json.load(open(args.golden, encoding="utf-8"))
    consultas_saida, problemas = [], []

    for c in golden["consultas"]:
        resposta = compila(c["resposta"])
        apoio = compila(c.get("apoio", []))
        graus_corpus, contagem = graus_do_acervo(trechos_norm, resposta, apoio)
        for e, n in zip(c["resposta"], contagem):
            if n == 0:
                problemas.append(f'{c["id"]}: o padrão não casa com nenhum trecho -> {e["arquivo"]} :: {e["padrao"]}')
        total_relevantes = sum(1 for g in graus_corpus if g == 2)
        if total_relevantes == 0:
            continue

        ranking = retriever.busca(c["pergunta"], k=K_MAX)           # [(fonte, texto)] em ordem de score
        idx_rank = [idx_por_objeto[id(t)] for t in ranking]
        graus_rank = [graus_corpus[i] for i in idx_rank]

        metricas = avaliar_consulta(graus_rank, total_relevantes, graus_corpus, ks)

        primeiro = next((p for p, g in enumerate(graus_rank, start=1) if g == 2), None)
        top_k_prod = set(idx_rank[:k_prod])
        resposta_no_contexto = any(graus_corpus[i] == 2 for i in (top_k_prod | idx_doc_fixo))

        consultas_saida.append({
            "id": c["id"],
            "categoria": c["categoria"],
            "pergunta": c["pergunta"],
            "total_relevantes": total_relevantes,
            "total_apoio": sum(1 for g in graus_corpus if g == 1),   # trechos de grau 1 no acervo (para o IDCG)
            "posicao_primeiro_relevante": primeiro,
            "resposta_no_contexto_real": resposta_no_contexto,   # top-k do app + documento fixo
            "precision_maxima": {k: min(total_relevantes, k) / k for k in ks},
            "metricas": metricas,
            "top": [{"posicao": p, "fonte": nome, "grau": g, "trecho": " ".join(texto.split())[:170]}
                    for p, ((nome, texto), g) in enumerate(zip(ranking, graus_rank), start=1)],
        })

    if problemas:
        print("PROBLEMAS NO GABARITO:")
        for p in problemas:
            print("  -", p)
        raise SystemExit("Corrija o golden_set.json antes de seguir.")

    resultados_k = [c["metricas"] for c in consultas_saida]
    agregado = agregar(resultados_k, ks)
    for k in ks:
        agregado[k]["precision_maxima"] = media(c["precision_maxima"][k] for c in consultas_saida)

    por_categoria = {}
    grupos = defaultdict(list)
    for c in consultas_saida:
        grupos[c["categoria"]].append(c)
    for cat, lista in grupos.items():
        agg = agregar([c["metricas"] for c in lista], ks)
        por_categoria[cat] = {"n": len(lista), "por_k": agg}

    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "recuperador": "BM25Retriever (rag.py) - chunk_size=1500, overlap=200, k1=1.5, b=0.75",
        "k_producao": k_prod,
        "ks": ks,
        "n_consultas": len(consultas_saida),
        "n_trechos_indexados": len(retriever.trechos),
        "n_documentos": len(docs),
        "agregado": agregado,
        "por_categoria": por_categoria,
        "cobertura_contexto_real": media(1.0 if c["resposta_no_contexto_real"] else 0.0 for c in consultas_saida),
        "criterio": golden.get("criterio_de_relevancia", {}),
        "aviso_gabarito": golden.get("aviso", ""),
        "consultas": consultas_saida,
    }
    RESULTADOS.mkdir(exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)

    a = agregado[k_prod]
    print(f"{len(consultas_saida)} perguntas | {len(retriever.trechos)} trechos indexados | k de produção = {k_prod}")
    print(f"  Precision@{k_prod} = {a['precision']:.3f} (máximo possível {a['precision_maxima']:.3f})")
    print(f"  Recall@{k_prod}    = {a['recall']:.3f}")
    print(f"  MRR@{k_prod}       = {a['mrr']:.3f}")
    print(f"  MAP@{k_prod}       = {a['map']:.3f}")
    print(f"  NDCG@{k_prod}      = {a['ndcg']:.3f}")
    print(f"  Resposta presente no contexto real (top-{k_prod} + documento fixo): {saida['cobertura_contexto_real']:.1%}")
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
