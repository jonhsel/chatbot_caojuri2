"""
Gera o relatório visual (HTML autocontido) a partir dos resultados.

Uso:
    python avaliacao_rag/gerar_relatorio.py
    python avaliacao_rag/gerar_relatorio.py --fragmento caminho.html   # também grava a versão sem <html>/<body>

Entradas:  resultados/retrieval.json, resultados/tokens.json, diagnostico.json
Saída:     resultados/relatorio_rag.html (abre no navegador, sem internet; as fontes são opcionais)
"""
import argparse
import json
import subprocess
from datetime import date

from comum import PASTA, RESULTADOS
from metricas_recuperacao import EXPLICACOES

ENVELOPE = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>{}</body></html>')


def referencia_git():
    """Commit, branch e data atuais, para o diagnóstico não ficar preso a um commit fixo."""
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=PASTA, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "desconhecido"
    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("branch", "--show-current"),
            "data": date.today().strftime("%d/%m/%Y")}


def montar_fragmento():
    dados = {
        "tokens": json.load(open(RESULTADOS / "tokens.json", encoding="utf-8")),
        "ret": json.load(open(RESULTADOS / "retrieval.json", encoding="utf-8")),
        "diag": json.load(open(PASTA / "diagnostico.json", encoding="utf-8")),
        "expl": EXPLICACOES,
    }
    dados["diag"]["referencia"] = dados["diag"]["referencia"].format(**referencia_git())
    modelo = (PASTA / "template_relatorio.html").read_text(encoding="utf-8")
    payload = json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")
    return modelo.replace("__DADOS__", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragmento", default=None)
    args = ap.parse_args()
    frag = montar_fragmento()
    saida = RESULTADOS / "relatorio_rag.html"
    saida.write_text(ENVELOPE.format(frag), encoding="utf-8")
    if args.fragmento:
        open(args.fragmento, "w", encoding="utf-8").write(frag)
    print(f"-> {saida} ({saida.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
