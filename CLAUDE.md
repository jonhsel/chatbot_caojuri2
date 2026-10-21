# CLAUDE.md

Chatbot "PARQUET" do CAOJÚRI (MPMA), feito em Streamlit + LangChain + Gemini.

## Arquivos principais

- `chatbot_caojuri_salvar.py`: app Streamlit em uso (chat, sidebar, salvar conversa).
- `chatbot_caojuri.py`: versão anterior, sem a opção de salvar conversa. Obsoleta; foi movida para `arquivos/chatbot_caojuri.py` (commit `8fc982a`).
- `rag.py`: recuperação de trechos (BM25) usada pelo app.
- `loaders.py`: loaders de PDF/CSV/TXT/MD (não usados no fluxo atual).
- `style.css`: estilos injetados no app.
- `avaliacao_rag/`: scripts e relatório de avaliação do RAG (ver seção abaixo).
- `Painel RAG CAOJÚRI.md`: texto do painel/relatório "antes e depois do RAG".
- `avaliacao_rag.zip`: cópia compactada de `avaliacao_rag/`.
- `arquivos/`: base de conhecimento em `.md`. O app baixa os arquivos do GitHub (`raw.githubusercontent.com/jonhsel/chatbot_caojuri2/master/arquivos/`), então **alterações em `arquivos/` só valem após push**.

A chave da API fica em `st.secrets["google"]["api_key"]`.

## Como funciona o fluxo

1. Botão "Iniciar o Parquet" → `carrega_arquivos` baixa os documentos (lista de `(nome, conteudo)`) → `carrega_modelo` cria a chain, o `BM25Retriever` e guarda tudo em `st.session_state` (`chain`, `retriever`, `doc_fixo`).
2. A cada pergunta, `pagina_chat` recupera os trechos mais relevantes e os passa ao modelo na variável `{contexto}` do system prompt.

## Como o sistema era antes das mudanças

Estado de `chatbot_caojuri_salvar.py` em 2026-10-20, antes da correção do 429 (o arquivo já tinha alterações não commitadas em relação ao último commit `93ceaf2`, então o `git diff` mostra só parte do que mudou).

**Carga dos documentos**

- `carrega_arquivos` baixava os 16 `.md` do GitHub, um por requisição HTTP, a cada clique em "Iniciar o Parquet".
- Devolvia **uma única string**, com os documentos concatenados por `\n\n`, sem separar por arquivo.

**Modelo e prompt**

- `carrega_modelo` montava o system prompt com uma **f-string** que embutia a string inteira dos documentos (`{documentos}`): ~1,7 MB, ~500 mil tokens.
- Modelo: `ChatGoogleGenerativeAI(model='gemini-3.8-flash', temperature=0.5, convert_system_message_to_human=True)`, com o `max_retries` padrão da biblioteca.
- Chain: `ChatPromptTemplate(system, placeholder chat_history, user) | chat`, guardada em `st.session_state['chain']`.

**Fluxo de cada pergunta (`pagina_chat`)**

1. `memoria.chat_memory.add_user_message(input)` gravava a pergunta **antes** de chamar o modelo.
2. `chain.stream({'input': ..., 'chat_history': memoria.buffer_as_messages})` enviava o **histórico inteiro**, que já incluía a pergunta atual, então ela ia duplicada (no histórico e em `input`).
3. Qualquer exceção (inclusive o 429) subia sem tratamento e o Streamlit mostrava o traceback.
4. Se a chamada falhasse, a pergunta ficava gravada na memória sem resposta.

**Consequências observadas**

- Cada pergunta consumia a cota de entrada do free tier (250 mil tokens/min) sozinha, e o app avisava aos usuários: "nosso chat processa até 2 perguntas por minuto".
- O erro `ResourceExhausted: 429 ... generate_content_free_tier_input_token_count, limit: 250000` aparecia como traceback.
- Documentos com chaves `{}` no texto podiam quebrar o template.

**CSS (`style.css`)**

- A mensagem da IA tinha fundo claro fixo (`rgb(230, 240, 250)`, via classe `st-emotion-cache-4oy321`) sem cor de texto definida. No tema dark o texto herdava a cor clara e ficava ilegível.

**Para consultar o código antigo:** `git show 93ceaf2:chatbot_caojuri_salvar.py` (versão do último commit, anterior também às alterações locais que já existiam).

## Histórico de mudanças

### Correção do erro 429 (cota do Gemini)

**Problema:** todos os documentos (~1,7 MB, ~500 mil tokens) eram enviados no system prompt a cada pergunta, mais o histórico inteiro. Isso estourava o limite do free tier (250 mil tokens de entrada por minuto) logo na primeira ou segunda pergunta.

**Mudanças:**

- **RAG com BM25 (`rag.py`):** os documentos são divididos em trechos de ~1500 caracteres (overlap 200) e indexados. Cada pergunta recupera os `K_TRECHOS` (8) mais relevantes, cerca de 2,5 mil tokens. Escolhi BM25 em vez de embeddings do Google porque indexar toda a base consumiria a cota de embeddings na inicialização, e BM25 não precisa de API nem de dependência nova.
- **Consulta com contexto:** a busca usa a pergunta anterior junto com a atual, para que perguntas de seguimento funcionem.
- **Documento fixo:** `CAOJÚRI - MPMA.md` (`DOC_SEMPRE_INCLUIDO`) vai sempre no contexto, por causa da regra 4 do prompt.
- **Janela de histórico:** só as últimas `JANELA_HISTORICO` (6) mensagens vão ao modelo. A tela continua exibindo a conversa completa.
- **Tratamento do 429:** `ResourceExhausted` (ou erro contendo "429") mostra um aviso com o tempo de espera extraído da mensagem; outros erros também são exibidos sem traceback. `max_retries=1` no `ChatGoogleGenerativeAI` evita esperas longas.
- **Bugs corrigidos:**
  - A pergunta era enviada duas vezes (no histórico e em `input`). Agora o histórico só recebe pergunta e resposta depois que a chamada dá certo; se falhar, nada é gravado.
  - O contexto entra como variável do template, e não via f-string. Antes, chaves `{}` dentro dos documentos podiam quebrar o `ChatPromptTemplate`.
  - Removido `convert_system_message_to_human`.

**Limites conhecidos:**

- BM25 casa palavras, não significado. Perguntas com vocabulário muito diferente do texto podem recuperar trechos fracos. Se isso ocorrer, o próximo passo é busca híbrida, com embeddings salvos em disco para não repetir a indexação.
- Testado apenas a recuperação em 10 dos 16 arquivos; o fluxo completo no Streamlit e a chamada ao Gemini não foram testados.
- Os "Comentários 1ª/2ª Fase" estão em `arquivos/`, mas fora da lista de `carrega_arquivos`. Com o RAG, incluí-los passa a ser viável.

### Contraste no tema dark

A resposta da IA tem fundo claro fixo em `style.css`, e no tema dark o texto também ficava claro. Foi adicionada no fim de `style.css` uma regra que força texto escuro (`#100530`) e links azuis (`#0b4fb3`) na mensagem da IA. Usa `data-testid` do Streamlit (mais estável que as classes `st-emotion-cache-*`). Não foi conferida visualmente no navegador.

### Ajustes no prompt e na interface (commits `60409ba` e `8fc982a`, 2026-10-21)

- **System prompt:** removida a regra que mandava buscar informações sobre o CAOJÚRI "apenas do documento CAOJÚRI-MPMA.md". As regras foram renumeradas: a 4 agora trata de temas fora do escopo (pedir para reformular) e a 5, de links (informar só os existentes na base). O prompt termina com "Abaixo estão os trechos mais relevantes dos documentos para a pergunta atual", seguido de `{contexto}`.
- **Aviso de 2 perguntas por minuto:** o `st.info` foi comentado em `pagina_chat`, já que o RAG eliminou o problema de cota.
- **`chatbot_caojuri.py`** (versão sem salvar conversa) foi movido para `arquivos/`.

### Avaliação do RAG (commit `0baaf9b`, 2026-10-21)

Pasta `avaliacao_rag/` (detalhes no `README.md` dela). Mede o consumo de tokens antes e depois do RAG e a qualidade da recuperação (Precision@k, Recall@k, MRR, MAP e NDCG).

- Rodar tudo: `python avaliacao_rag/executar_tudo.py` (opcional: `--tpm`, `--rpm`, `--rpd` com os limites reais da chave). Dependências extras: `pip install google-genai sentencepiece`.
- Resultado: `avaliacao_rag/resultados/relatorio_rag.html`, ou `streamlit run avaliacao_rag/painel_metricas.py`.
- `golden_set.json`: 51 perguntas com gabarito (padrões de texto). Editar com perguntas reais dos usuários melhora a confiança das métricas.
- Os scripts leem arquivos, `K_TRECHOS`, janela de histórico, modelo e prompt direto de `chatbot_caojuri_salvar.py`, então acompanham mudanças no app.
- `medir_tokens.py --modo api` (contagem exata pelo Gemini) não foi testado por falta de chave. TPM/RPM/RPD não são publicados por modelo pelo Google; o TPM padrão é uma premissa.

## Ajustes rápidos

- Mais ou menos contexto por pergunta: `K_TRECHOS`, no topo de `chatbot_caojuri_salvar.py`.
- Memória mais longa ou curta: `JANELA_HISTORICO`, no mesmo lugar.
- Tamanho dos trechos: `chunk_size` e `chunk_overlap` em `BM25Retriever` (`rag.py`).
- Para adicionar um documento à base, inclua o `.md` em `arquivos/`, faça push e adicione o nome à lista em `carrega_arquivos`.
