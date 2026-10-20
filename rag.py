import math
import re
import unicodedata
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

STOPWORDS = set('''
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre
e ou mas que se ao aos à às pelo pela pelos pelas como qual quais quando onde quem
é são foi ser ter tem há esta este isto esse essa isso seu sua seus suas me te lhe nos
mais menos muito muita já não sim também então só até entre sobre ainda pode podem
'''.split())


def _sem_acento(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


def tokeniza(texto):
    texto = _sem_acento(texto.lower())
    tokens = []
    for t in re.findall(r'\w+', texto):
        if t in STOPWORDS or _sem_acento(t) in STOPWORDS:
            continue
        # stemming leve: remove plural simples
        if len(t) > 4 and t.endswith('s'):
            t = t[:-1]
        tokens.append(t)
    return tokens


class BM25Retriever:
    """Busca lexical BM25 sobre trechos dos documentos. Não usa API externa (sem cota)."""

    def __init__(self, documentos, chunk_size=1500, chunk_overlap=200, k1=1.5, b=0.75):
        """documentos: lista de (nome_arquivo, conteudo)."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.trechos = []  # (fonte, texto)
        for nome, conteudo in documentos:
            for trecho in splitter.split_text(conteudo):
                # o nome do arquivo entra na indexação para casar perguntas sobre o documento
                self.trechos.append((nome, trecho))

        self.k1, self.b = k1, b
        self.tf = [Counter(tokeniza(f'{nome} {texto}')) for nome, texto in self.trechos]
        self.tamanhos = [sum(c.values()) for c in self.tf]
        self.media = (sum(self.tamanhos) / len(self.tamanhos)) if self.tamanhos else 0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def busca(self, consulta, k=8):
        termos = [t for t in set(tokeniza(consulta)) if t in self.idf]
        pontuacoes = []
        for i, c in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.tamanhos[i] / self.media)
            for t in termos:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                pontuacoes.append((s, i))
        pontuacoes.sort(reverse=True)
        return [self.trechos[i] for _, i in pontuacoes[:k]]

    def contexto(self, consulta, k=8):
        """Texto formatado com os k trechos mais relevantes."""
        return '\n\n'.join(f'[Fonte: {nome}]\n{texto}' for nome, texto in self.busca(consulta, k))
