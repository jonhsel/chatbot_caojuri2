# Avaliação do RAG do chatbot CAOJÚRI

Mede o consumo de tokens antes e depois do RAG e a qualidade da recuperação (Precision@k, Recall@k, MRR, MAP e NDCG).

## Passo a passo

1. Instale as dependências extras (o resto já está no `requirements.txt`):
   ```
   pip install google-genai sentencepiece
   ```
2. Rode tudo de uma vez, a partir da raiz do projeto:
   ```
   python avaliacao_rag/executar_tudo.py
   ```
   Os limites (TPM, RPM, RPD) vêm de `limites_modelos.json`, copiados do painel *Rate limit* do AI Studio. Para outro modelo ou nível:
   ```
   python avaliacao_rag/executar_tudo.py --modelo gemini-3.8-flash
   python avaliacao_rag/executar_tudo.py --tpm 4000000 --rpm 1000 --rpd 10000
   ```
3. Veja o resultado:
   - abra `avaliacao_rag/resultados/relatorio_rag.html` no navegador, ou
   - `streamlit run avaliacao_rag/painel_metricas.py` para ver dentro do Streamlit.

## O que cada arquivo faz

| Arquivo | Função |
|---|---|
| `metricas_recuperacao.py` | As cinco métricas (funções puras, com explicações) |
| `test_metricas.py` | Testes das métricas contra valores calculados à mão |
| `golden_set.json` | 51 perguntas com o gabarito (padrões de texto) |
| `avaliar_retrieval.py` | Roda o `BM25Retriever` do app e calcula as métricas |
| `medir_tokens.py` | Tokens antes e depois do RAG e comparação com TPM, RPM, RPD e janela |
| `limites_modelos.json` | TPM, RPM e RPD por modelo, do painel do AI Studio |
| `gerar_relatorio.py` | Monta o relatório HTML a partir dos resultados |
| `painel_metricas.py` | Página Streamlit que mostra o relatório |
| `diagnostico.json` | Achados da análise do projeto (texto editável) |

Os scripts leem a configuração (arquivos, `K_TRECHOS`, janela de histórico, modelo, prompt) direto de `chatbot_caojuri_salvar.py`, então acompanham o código.

## Como melhorar a confiança dos números

- **Gabarito:** edite `golden_set.json` com perguntas reais dos usuários. Cada pergunta tem `resposta` (trechos que contêm a resposta) e, opcionalmente, `apoio` (contexto útil). O avaliador recusa padrões que não casam com nenhum trecho.
- **Tokens exatos:** `python avaliacao_rag/medir_tokens.py --modo api` usa a contagem do próprio Gemini (exige `GOOGLE_API_KEY`). Esse modo não foi testado na análise original, por falta de chave.
- **Limites:** edite `limites_modelos.json` quando mudar de nível ou de modelo. O limite de entrada do Gemini 3.5 Flash Lite ainda não foi confirmado (o do 3.8 Flash, 1.048.576, vem da página do modelo).
