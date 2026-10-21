"""
Utilidades compartilhadas pelos scripts de avaliação.

Princípio: nada é copiado à mão do app. A configuração (arquivos indexados,
K_TRECHOS, janela de histórico, modelo, mensagem de sistema) é LIDA do
chatbot_caojuri_salvar.py, para que as medições sigam o código real.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent
RESULTADOS = PASTA / "resultados"
APP = RAIZ / "chatbot_caojuri_salvar.py"

# permite `from rag import BM25Retriever` (o rag.py fica na raiz do projeto)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------

def nfc(texto: str) -> str:
    """Normaliza Unicode (nomes de arquivo podem vir decompostos, ex.: 'ç' = 'c' + cedilha)."""
    return unicodedata.normalize("NFC", texto)


def normaliza(texto: str) -> str:
    """minúsculas, sem acentos e com espaços/quebras colapsados (os .md têm espaços duplos)."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto)


# --------------------------------------------------------------------------
# Configuração do app
# --------------------------------------------------------------------------

def ler_config_do_app() -> dict:
    src = APP.read_text(encoding="utf-8")
    bloco = src.split("arquivos = [", 1)[1].split("]", 1)[0]
    return {
        "k_trechos": int(re.search(r"^K_TRECHOS\s*=\s*(\d+)", src, re.M).group(1)),
        "janela_historico": int(re.search(r"^JANELA_HISTORICO\s*=\s*(\d+)", src, re.M).group(1)),
        "doc_fixo": re.search(r'^DOC_SEMPRE_INCLUIDO\s*=\s*"([^"]+)"', src, re.M).group(1),
        "arquivos": re.findall(r'"([^"]+\.md)"', bloco),
        "modelo": re.search(r"ChatGoogleGenerativeAI\(\s*model='([^']+)'", src).group(1),
        "system_message": re.search(r"system_message = '''(.*?)'''", src, re.S).group(1),
    }


# Único trecho que o commit do RAG alterou na mensagem de sistema (commit 46b6872):
_FRASE_RAG = "    Abaixo estão os trechos mais relevantes dos documentos para a pergunta atual:\n"


def system_message_antes_do_rag(system_message_depois: str) -> str:
    """Mensagem de sistema da versão anterior ao RAG (documentos inteiros no lugar do contexto)."""
    return system_message_depois.replace(_FRASE_RAG, "").replace("{contexto}", "{documentos}")


# --------------------------------------------------------------------------
# Documentos
# --------------------------------------------------------------------------

def carregar_documentos(nomes: list[str], pasta: Path | None = None) -> list[tuple[str, str]]:
    """Lê os .md da pasta 'arquivos' (mesmo conteúdo que o app baixa do GitHub/master).

    Devolve [(nome_como_no_app, conteudo)], a mesma estrutura que o app passa ao BM25Retriever.
    """
    pasta = pasta or (RAIZ / "arquivos")
    reais = {nfc(f): f for f in os.listdir(pasta)}
    docs = []
    for nome in nomes:
        real = reais.get(nfc(nome))
        if real is None:
            raise FileNotFoundError(f"Arquivo da lista do app não encontrado em {pasta}: {nome}")
        docs.append((nome, (pasta / real).read_text(encoding="utf-8")))
    return docs


# --------------------------------------------------------------------------
# Contagem de tokens
# --------------------------------------------------------------------------

class ContadorTokens:
    """Conta tokens de um texto.

    modos:
      'local'      tokenizador SentencePiece da família Gemma/Gemini (via google-genai).
                   Offline depois do 1º uso, exato para a família 2.x e uma boa aproximação
                   para modelos mais novos.
      'api'        endpoint count_tokens do Gemini (exato para o modelo informado; exige chave
                   em GOOGLE_API_KEY ou --api-key).
      'estimativa' caracteres / 3,6 (último recurso).
      'auto'       tenta 'api' se houver chave, senão 'local', senão 'estimativa'.
    """

    def __init__(self, modo: str = "auto", api_key: str | None = None, modelo: str = "gemini-3.8-flash"):
        self.modelo = modelo
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._fn = None
        ordem = {"auto": ["api", "local", "estimativa"]}.get(modo, [modo])
        erros = []
        for tentativa in ordem:
            if tentativa == "api" and not api_key:
                continue
            try:
                self._fn, self.descricao = self._criar(tentativa, api_key)
                self.modo = tentativa
                break
            except Exception as e:  # noqa: BLE001 - queremos cair para o próximo modo
                erros.append(f"{tentativa}: {type(e).__name__}: {str(e)[:120]}")
        if self._fn is None:
            raise RuntimeError("Nenhum contador de tokens disponível. " + " | ".join(erros))
        if self.modo == "estimativa":
            print("AVISO: usando estimativa por caracteres; os números de tokens são aproximados.")

    def _criar(self, modo, api_key):
        if modo == "api":
            from google import genai
            client = genai.Client(api_key=api_key)
            modelo = self.modelo
            client.models.count_tokens(model=modelo, contents="teste")  # falha cedo se a chave/modelo não servir
            return (lambda t: client.models.count_tokens(model=modelo, contents=t).total_tokens,
                    f"API count_tokens ({modelo}) - contagem exata")
        if modo == "local":
            warnings.filterwarnings("ignore")
            from google.genai import local_tokenizer
            tk = local_tokenizer.LocalTokenizer(model_name="gemini-2.5-flash")
            return (lambda t: tk.count_tokens(t).total_tokens if t else 0,
                    "Tokenizador SentencePiece Gemma/Gemini (local, via google-genai) - "
                    "aproximação do tokenizador do modelo-alvo")
        if modo == "estimativa":
            return (lambda t: int(len(t) / 3.6 + 0.5), "Estimativa: caracteres / 3,6")
        raise ValueError(f"modo desconhecido: {modo}")

    def __call__(self, texto: str) -> int:
        return int(self._fn(texto))
