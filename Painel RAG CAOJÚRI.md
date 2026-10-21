# Chatbot CAOJÚRI: antes e depois do RAG

Consumo de tokens, limites do modelo e qualidade da recuperação, medidos sobre o código e os 16 documentos do projeto.

[Resumo](#resumo)[Antes e depois](#arquitetura)[Tokens e limites](#tokens) [Métricas de recuperação](#metricas)[Perguntas](#perguntas)[Diagnóstico](#diagnostico)[Como interpretar](#cuidados)

## Resumo

## Como era e como ficou

Mesmo modelo e mesmos documentos. A mudança está no que vai dentro de cada pergunta.

Antes · commit 93ceaf2

### O acervo inteiro em toda pergunta

- Os 16 documentos eram concatenados e colocados na mensagem de sistema.
- O histórico completo da conversa ia junto, crescendo a cada turno.
- Cada pergunta consumia quase metade da janela de contexto e do TPM por minuto.

Depois · commit 46b6872

### Só o que a pergunta precisa

- O rag.py divide os documentos em trechos de 1.500 caracteres e os indexa com BM25.
- A cada pergunta entram os trechos mais relevantes e o documento “CAOJÚRI - MPMA.md”.
- O histórico vai limitado às últimas mensagens.

### O que compõe o prompt depois do RAG (a partir do 4º turno)

## Consumo de tokens

Entrada por pergunta. A saída (resposta) é igual nos dois cenários.

### Tokens de entrada por turno de uma conversa (escala logarítmica)

### Onde estão os tokens do acervo

### Comparação com os limites do modelo premissas editáveis

A documentação do Google não publica TPM, RPM e RPD por modelo; eles aparecem no painel Rate limit do AI Studio, por projeto. O TPM inicial é uma premissa: é o valor que explica o aviso do próprio app (2 perguntas por minuto). Troque pelos seus.

TPM (tokens de entrada/min) RPM (requisições/min) RPD (requisições/dia) Limite de entrada (tokens)

## Qualidade da recuperação

### O que cada métrica diz

### Precision@k e Recall@k

### MRR, MAP e NDCG por k

### Por tipo de pergunta

### Um exemplo, passo a passo

## Pergunta por pergunta

Use esta tabela para conferir o gabarito: se um trecho marcado como ✗ na verdade responde, o gabarito precisa de ajuste, não o recuperador.

Tipo Situação

| ID | Pergunta | Tipo | 1º relevante | No contexto real |
| --- | --- | --- | --- | --- |

## Diagnóstico do projeto

### Próximos passos sugeridos

## Como interpretar estes números