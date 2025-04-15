#import docling.document_converter
import docling.document_converter
import streamlit as st
import tempfile
import os  # Importe o módulo os para lidar com arquivos e diretórios

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import shutil

from docling.document_converter import DocumentConverter
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

# Diretórios de entrada e saída
INPUT_DIR = './entrada'
OUTPUT_DIR = './arquivos'
FORMATO_CONVERSAO = 'md' # Formato Markdown

# Função para converter arquivos
def converter_arquivos(input_dir, output_dir, formato):
    """Converte arquivos para o formato especificado."""
    try:
        # Verifica se os diretórios de entrada e saída existem
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Diretório de entrada '{input_dir}' não encontrado.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Cria o diretório de saída, se não existir

        conversor = DocumentConverter()
        conversor.convert_all(input_dir, output_dir, formato)

        print(f"Todos os arquivos em '{input_dir}' foram convertidos para '{formato}' e salvos em '{output_dir}'.")

    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Ocorreu um erro durante a conversão: {e}")

# Função para carregar arquivos (adaptada para usar a pasta 'arquivos')
def carrega_arquivos(pasta_arquivos):
    documentos = []
    for nome_arquivo in os.listdir(pasta_arquivos):
        caminho_arquivo = os.path.join(pasta_arquivos, nome_arquivo)
        if os.path.isfile(caminho_arquivo):
            try:
                with open(caminho_arquivo, 'r', encoding='utf-8') as f: # Especifica a codificação UTF-8
                    conteudo = f.read()
                    documentos.append(conteudo)
            except Exception as e:
                st.error(f"Erro ao ler o arquivo '{nome_arquivo}': {e}")
    return documentos

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
        # Exibe os arquivos na pasta 'entrada'
        if os.path.exists(INPUT_DIR) and os.listdir(INPUT_DIR):
            st.write(f"Arquivos na pasta '{INPUT_DIR}':")
            for filename in os.listdir(INPUT_DIR):
                st.write(filename)
        elif not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR): # Verifica se a pasta de entrada está vazia
            st.warning(f"Nenhum arquivo encontrado na pasta '{INPUT_DIR}'.")

        # Exibe os arquivos na pasta 'arquivos' (após a conversão)
        if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
            st.write(f"Arquivos convertidos (Markdown) em '{OUTPUT_DIR}':")
            for filename in os.listdir(OUTPUT_DIR):
                st.write(filename)

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
        
        pasta_arquivos = OUTPUT_DIR
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
    # Cria os diretórios se não existirem
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Converte os arquivos automaticamente ao iniciar o Streamlit
    converter_arquivos(INPUT_DIR, OUTPUT_DIR, FORMATO_CONVERSAO)
    main()