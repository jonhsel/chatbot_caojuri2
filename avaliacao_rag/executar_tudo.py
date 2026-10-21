"""Roda as três etapas em ordem: tokens, recuperação e relatório.

    python avaliacao_rag/executar_tudo.py [argumentos de medir_tokens.py, ex.: --tpm 4000000 --rpm 1000]
"""
import subprocess
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
extra = sys.argv[1:]

for script, args in (("test_metricas.py", []), ("medir_tokens.py", extra), ("avaliar_retrieval.py", []), ("gerar_relatorio.py", [])):
    print(f"\n=== {script} ===")
    if subprocess.run([sys.executable, str(PASTA / script), *args], cwd=PASTA).returncode != 0:
        sys.exit(f"Falhou em {script}")
print("\nPronto. Abra avaliacao_rag/resultados/relatorio_rag.html ou rode: streamlit run avaliacao_rag/painel_metricas.py")
