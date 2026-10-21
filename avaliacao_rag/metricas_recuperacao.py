"""
Métricas de recuperação de informação (Precision@k, Recall@k, MRR, MAP e NDCG).

Funções puras, sem dependências externas, para avaliar qualquer recuperador
(BM25, busca vetorial, híbrido...). Todas recebem a lista de graus de
relevância dos resultados NA ORDEM EM QUE O RECUPERADOR OS DEVOLVEU
(posição 1 primeiro).

Convenções usadas neste projeto
-------------------------------
- Relevância binária (Precision, Recall, MRR, MAP): 1 = o trecho contém a
  resposta; 0 = não contém.
- Relevância graduada (NDCG): 2 = o trecho contém a resposta; 1 = o trecho é
  contexto útil (mesmo documento, assunto relacionado); 0 = irrelevante.
- "total_relevantes" é quantos trechos relevantes existem NO ACERVO INTEIRO
  para aquela pergunta (não apenas entre os k devolvidos). É ele que permite
  calcular Recall e normalizar o AP.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# Métricas por pergunta
# --------------------------------------------------------------------------

def precision_at_k(relevancias: Sequence[int], k: int) -> float:
    """Dos k primeiros resultados, que fração é relevante?

    Divide sempre por k: se o recuperador devolveu menos de k itens, os
    "buracos" contam como não relevantes.
    """
    if k <= 0:
        raise ValueError("k deve ser positivo")
    return sum(1 for r in relevancias[:k] if r > 0) / k


def recall_at_k(relevancias: Sequence[int], k: int, total_relevantes: int) -> float:
    """De todos os trechos relevantes que existem, que fração apareceu nos k primeiros?"""
    if total_relevantes <= 0:
        return 0.0
    return sum(1 for r in relevancias[:k] if r > 0) / total_relevantes


def reciprocal_rank(relevancias: Sequence[int], k: int | None = None) -> float:
    """1 / posição do primeiro resultado relevante (0 se não houver nos k primeiros).

    A média desse valor sobre várias perguntas é o MRR.
    """
    topo = relevancias if k is None else relevancias[:k]
    for posicao, r in enumerate(topo, start=1):
        if r > 0:
            return 1.0 / posicao
    return 0.0


def average_precision_at_k(relevancias: Sequence[int], k: int, total_relevantes: int) -> float:
    """Precisão média até k (AP@k). A média sobre perguntas é o MAP@k.

    AP@k = (1 / min(R, k)) * soma, para cada posição i <= k com resultado
    relevante, de Precision@i. Normalizar por min(R, k) faz com que um
    ranking perfeito sempre valha 1,0, mesmo quando R > k.
    """
    if total_relevantes <= 0:
        return 0.0
    acertos = 0
    soma = 0.0
    for posicao, r in enumerate(relevancias[:k], start=1):
        if r > 0:
            acertos += 1
            soma += acertos / posicao
    return soma / min(total_relevantes, k)


def dcg_at_k(graus: Sequence[int], k: int) -> float:
    """Ganho acumulado descontado: ganho (2^grau - 1) / log2(posição + 1)."""
    return sum((2 ** g - 1) / math.log2(posicao + 1)
               for posicao, g in enumerate(graus[:k], start=1))


def ndcg_at_k(graus: Sequence[int], k: int, graus_ideais: Iterable[int]) -> float:
    """DCG normalizado pelo melhor DCG possível (ordem ideal de todos os graus do acervo)."""
    ideal = sorted((g for g in graus_ideais if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(graus, k) / idcg


# --------------------------------------------------------------------------
# Agregação
# --------------------------------------------------------------------------

def avaliar_consulta(graus: Sequence[int], total_relevantes: int,
                     graus_ideais: Iterable[int], ks: Sequence[int]) -> dict:
    """Calcula todas as métricas de UMA pergunta para vários valores de k.

    graus: grau (0/1/2) de cada resultado devolvido, na ordem do ranking.
    O grau 2 é o que conta como "relevante" nas métricas binárias.
    """
    binarias = [1 if g >= 2 else 0 for g in graus]
    graus_ideais = list(graus_ideais)
    saida = {}
    for k in ks:
        saida[k] = {
            "precision": precision_at_k(binarias, k),
            "recall": recall_at_k(binarias, k, total_relevantes),
            "rr": reciprocal_rank(binarias, k),
            "ap": average_precision_at_k(binarias, k, total_relevantes),
            "ndcg": ndcg_at_k(graus, k, graus_ideais),
        }
    return saida


def media(valores: Iterable[float]) -> float:
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0


def agregar(resultados_por_consulta: Sequence[dict], ks: Sequence[int]) -> dict:
    """Médias sobre as perguntas: Precision@k, Recall@k, MRR@k, MAP@k e NDCG@k."""
    agregado = {}
    for k in ks:
        agregado[k] = {
            "precision": media(r[k]["precision"] for r in resultados_por_consulta),
            "recall": media(r[k]["recall"] for r in resultados_por_consulta),
            "mrr": media(r[k]["rr"] for r in resultados_por_consulta),
            "map": media(r[k]["ap"] for r in resultados_por_consulta),
            "ndcg": media(r[k]["ndcg"] for r in resultados_por_consulta),
        }
    return agregado


# --------------------------------------------------------------------------
# Textos explicativos (usados pelo relatório visual)
# --------------------------------------------------------------------------

EXPLICACOES = {
    "precision": {
        "nome": "Precision@k",
        "pergunta": "Dos k trechos que o chatbot recebeu, quantos realmente servem?",
        "formula": "Precision@k = (nº de trechos relevantes nos k primeiros) ÷ k",
        "como_ler": "Mede o RUÍDO no contexto. Com k = 8 e 1 trecho útil, vale 0,125: "
                    "isso é esperado quando só existe um trecho com a resposta. "
                    "Compare sempre com o máximo possível (1 ÷ k).",
        "limite": "Não liga para a ordem dos trechos nem para quantos relevantes existem no acervo.",
    },
    "recall": {
        "nome": "Recall@k",
        "pergunta": "De tudo que continha a resposta, quanto chegou ao chatbot?",
        "formula": "Recall@k = (nº de relevantes nos k primeiros) ÷ (nº total de relevantes no acervo)",
        "como_ler": "É a métrica mais importante para RAG: se o trecho com a resposta não "
                    "entra no contexto, o modelo não tem como acertar (ou inventa). "
                    "Recall@8 = 0,80 significa que em 80% das perguntas a resposta chegou ao modelo.",
        "limite": "Não pune ruído: devolver 100 trechos tem recall alto, mas encarece e confunde o modelo.",
    },
    "mrr": {
        "nome": "MRR (Mean Reciprocal Rank)",
        "pergunta": "Em que posição aparece o PRIMEIRO trecho útil?",
        "formula": "RR = 1 ÷ posição do 1º relevante;  MRR = média do RR sobre todas as perguntas",
        "como_ler": "1,00 = o 1º resultado sempre serve; 0,50 = em média na 2ª posição; "
                    "0,33 = na 3ª. Importa quando o modelo dá mais peso ao começo do contexto.",
        "limite": "Só olha o primeiro acerto; ignora os demais relevantes.",
    },
    "map": {
        "nome": "MAP (Mean Average Precision)",
        "pergunta": "Os trechos úteis estão concentrados no topo do ranking?",
        "formula": "AP@k = (soma de Precision@i nas posições i com acerto) ÷ min(R, k);  MAP = média do AP",
        "como_ler": "Combina precisão e ordem: premia quem coloca TODOS os relevantes no topo. "
                    "Vale 1,00 só se todos os relevantes vierem antes de qualquer irrelevante.",
        "limite": "Usa relevância binária (serve / não serve).",
    },
    "ndcg": {
        "nome": "NDCG@k (Normalized Discounted Cumulative Gain)",
        "pergunta": "A ordem está próxima da ideal, considerando que há trechos mais e menos úteis?",
        "formula": "DCG@k = Σ (2^grau − 1) ÷ log2(posição + 1);  NDCG = DCG ÷ DCG do ranking ideal",
        "como_ler": "Aceita graus de utilidade (2 = tem a resposta, 1 = contexto útil, 0 = inútil) e "
                    "desconta acertos que aparecem mais embaixo. 1,00 = ordem ideal.",
        "limite": "Depende de quem define os graus; aqui são definidos por padrões de texto (ver golden_set.json).",
    },
}
