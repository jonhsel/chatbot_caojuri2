"""
Painel de métricas do RAG dentro do Streamlit.

Rodar:   streamlit run avaliacao_rag/painel_metricas.py
Mostra o relatório gerado por gerar_relatorio.py. Se os resultados ainda não existirem,
o botão abaixo executa as três etapas (leva poucos segundos).
"""
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PASTA = Path(__file__).resolve().parent
RELATORIO = PASTA / "resultados" / "relatorio_rag.html"

st.set_page_config(page_title="Painel RAG CAOJÚRI", layout="wide")

if st.button("Recalcular métricas e relatório"):
    with st.spinner("Medindo tokens, avaliando a recuperação e gerando o relatório..."):
        for script in ("medir_tokens.py", "avaliar_retrieval.py", "gerar_relatorio.py"):
            r = subprocess.run([sys.executable, str(PASTA / script)], cwd=PASTA, capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"Falha em {script}:\n{r.stdout}\n{r.stderr}")
                st.stop()
    st.success("Relatório atualizado.")

if RELATORIO.exists():
    components.html(RELATORIO.read_text(encoding="utf-8"), height=4200, scrolling=True)
else:
    st.info("Ainda não há relatório. Clique em “Recalcular métricas e relatório”.")
