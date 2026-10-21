"""
Testes das métricas contra valores calculados à mão.
Execute:  python avaliacao_rag/test_metricas.py   (ou: pytest avaliacao_rag/test_metricas.py)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metricas_recuperacao import (average_precision_at_k, avaliar_consulta, dcg_at_k,
                                  ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank)


def aprox(a, b, tol=1e-9):
    return abs(a - b) < tol


# Ranking de exemplo: posições 1 e 3 são relevantes; existem 3 relevantes no acervo.
RANK = [1, 0, 1, 0, 0]
R = 3


def test_precision():
    assert aprox(precision_at_k(RANK, 1), 1.0)
    assert aprox(precision_at_k(RANK, 3), 2 / 3)
    assert aprox(precision_at_k(RANK, 5), 2 / 5)
    # menos resultados que k: os "buracos" contam como erro
    assert aprox(precision_at_k([1], 4), 1 / 4)


def test_recall():
    assert aprox(recall_at_k(RANK, 1, R), 1 / 3)
    assert aprox(recall_at_k(RANK, 3, R), 2 / 3)
    assert recall_at_k(RANK, 5, 0) == 0.0


def test_rr():
    assert aprox(reciprocal_rank(RANK), 1.0)
    assert aprox(reciprocal_rank([0, 0, 1]), 1 / 3)
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert reciprocal_rank([0, 0, 1], k=2) == 0.0  # fora do corte


def test_ap():
    # (P@1 + P@3) / min(3, 5) = (1 + 2/3) / 3
    assert aprox(average_precision_at_k(RANK, 5, R), (1 + 2 / 3) / 3)
    # ranking perfeito com R > k continua valendo 1
    assert aprox(average_precision_at_k([1, 1], 2, 10), 1.0)


def test_ndcg_binario():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)                         # 1 + 0,5
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)     # 1 + 0,6309 + 0,5
    assert aprox(ndcg_at_k(RANK, 5, [1, 1, 1]), dcg / idcg)


def test_ndcg_graduado():
    graus = [1, 2, 0]                 # contexto útil primeiro, resposta depois
    ganhos = (2 ** 1 - 1) / math.log2(2) + (2 ** 2 - 1) / math.log2(3)
    ideal = (2 ** 2 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(3)
    assert aprox(ndcg_at_k(graus, 3, [2, 1]), ganhos / ideal)
    assert aprox(ndcg_at_k([2, 1, 0], 3, [2, 1]), 1.0)   # ordem ideal = 1


def test_avaliar_consulta_integra():
    graus = [0, 2, 1, 0]
    saida = avaliar_consulta(graus, total_relevantes=1, graus_ideais=[2, 1], ks=[1, 3])
    assert saida[1]["precision"] == 0.0 and saida[1]["rr"] == 0.0
    assert aprox(saida[3]["rr"], 0.5)
    assert aprox(saida[3]["recall"], 1.0)
    assert aprox(saida[3]["ap"], 0.5)          # P@2 = 1/2, dividido por min(1,3)=1


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
            print("ok ", nome)
    print("Todos os testes passaram.")
