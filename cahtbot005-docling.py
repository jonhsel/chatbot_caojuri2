import streamlit as st
import tempfile
import os  # Importe o módulo os para lidar com arquivos e diretórios

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import docling

# Importe as funções de carregamento de arquivos
#from loaders import carrega_pdf, carrega_csv, carrega_txt, carrega_site, carrega_youtube

#===============
#CSS

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

#################

st.image('images/chatbot2.png')

# Remova a seleção manual de tipo de arquivo
# TIPOS_ARQUIVOS = ['Arquivos .pdf', 'Site', 'Youtube', 'Arquivos .csv', 'Arquivos .txt']

CONFIG_MODELOS = {  'OpenAI': 
                            {'modelos': ['gpt-4o-mini', 'gpt-4o'],
                            'chat': ChatOpenAI}
}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(pasta_arquivos):
    documentos = []
    for nome_arquivo in os.listdir(pasta_arquivos):
        caminho_arquivo = os.path.join(pasta_arquivos, nome_arquivo)
        if os.path.isfile(caminho_arquivo):
            try:
                documento = docling.process_file(caminho_arquivo)
                #usa o docling para processar o arquivo                

                if documento: # Verifica se o documento foi carregado com sucesso
                    documentos.append(documento)

            except Exception as e:
                st.error(f"Erro ao carregar arquivo {nome_arquivo}: {e}")

    return "\n\n".join(documentos)  # Concatena todos os documentos com separadores

def carrega_modelo(provedor, modelo, api_key, documentos):

    if not documentos:  # Verifica se há documentos para carregar
        st.error("Nenhum documento foi carregado. Verifique a pasta 'arquivos'.")
        return

    system_message = f''' Você é o chatb.ot virtual do CAOJÚRI.
    Você possui acesso às seguintes informações vindas de um ou mais documentos:
    
    ####
    {documentos}
    ####
    Utilize apenas as informações fornecidas nos documentos para basear suas respostas.

   
    '''

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain


def pagina_chat():
    st.header('⚖️ Chat Virtual do CAOJÚRI')
    st.write('Minha função é responder questionamentos realcionados ao Tribunal do Júri!')

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('⚠️ Carregue a api e inicialize o assistente!')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Me pergunte algo!')
    if input_usuario:
        memoria.chat_memory.add_user_message(input_usuario)
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
            }))
        #resposta = chat_model.invoke(input_usuario).content
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria
        #st._rerun()
    # ... (resto do código da página de chat)

def sidebar():
    tabs_assistente = st.tabs(['Seleção de Arquivos', 'Modelo de IA'])
    with tabs_assistente[0]:
        pasta_arquivos = 'arquivos'  # Define a pasta de arquivos
        if not os.path.exists(pasta_arquivos) or not os.listdir(pasta_arquivos):
            st.warning(f"Nenhum arquivo encontrado na pasta '{pasta_arquivos}'.")
            return  # Sai da função se não houver arquivos na pasta

        st.write(f"Arquivos encontrados na pasta '{pasta_arquivos}':")
        for nome_arquivo in os.listdir(pasta_arquivos):
            st.write(nome_arquivo)  # Lista os arquivos na sidebar

    with tabs_assistente[1]:
        provedor = st.selectbox('Selecione a empresa criadora do modelo de IA', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo de IA', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API do modelo escolhido{provedor}',
            value=st.session_state.get(f'api_key_{provedor}')
        )
        st.session_state[f'api_key_{provedor}'] = api_key
        # ... (resto do código da aba de modelo de IA)

    if st.button('▶️ Iniciar o Assistente', use_container_width=True):
        documentos = carrega_arquivos(pasta_arquivos)  # Carrega todos os arquivos
        carrega_modelo(provedor, modelo, api_key, documentos)

    if st.button('️ Limpar o histórico de conversação', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    # ... (resto do código principal)

if __name__=='__main__':
    main()