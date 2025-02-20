import streamlit as st
import tempfile
import os  # Importe o módulo os para lidar com arquivos e diretórios

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Importe as funções de carregamento de arquivos
from loaders import carrega_pdf, carrega_csv, carrega_txt, carrega_site, carrega_youtube

#===============
#CSS

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

#################

st.image('images/juria.png')

# Remova a seleção manual de tipo de arquivo
# TIPOS_ARQUIVOS = ['Arquivos .pdf', 'Site', 'Youtube', 'Arquivos .csv', 'Arquivos .txt']

CONFIG_MODELOS = {  'OpenAI': 
                            {'modelos': ['gpt-4o-mini', 'gpt-4o'],
                            'chat': ChatOpenAI}
}

MEMORIA = ConversationBufferMemory()

def carrega_arquivo (nome_arquivo):
    # Extrai a extensão do arquivo
    _, extensao = os.path.splitext(nome_arquivo)
    extensao = extensao.lower()

    if extensao == '.pdf':
        documento = carrega_pdf(nome_arquivo)
    elif extensao == '.csv':
        documento = carrega_csv(nome_arquivo)
    elif extensao == '.txt':
        documento = carrega_txt(nome_arquivo)
    # Adapte para outros tipos de arquivo se necessário
    else:
        st.error(f"Tipo de arquivo não suportado: {extensao}")
        return None
    
    return documento

def carrega_modelo(provedor, modelo, api_key, nome_arquivo):
    documento = carrega_arquivo(nome_arquivo)

    if documento is None:
        return  # Sai da função se o carregamento falhar

    system_message = f''' Você é um assistente técnico chamado 'assistente do Jonh Selmo'.
    Você possui acesso às seguintes informações vindas de um documento:
    
    ####
    {documento}
    ####
    Utilize as informações fornecidas para basear suas respostas.

    Sempre que houver $ na saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and coockies to continue", sugira ao usuário carregar novamente o 'Assistente do Jonh Selmo'!
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
    st.header('⚖️ Assistente Virtual - CAOJÚRI')

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('⚠️ Carregue o arquivo ou digite a url/id antes de inicializar o assistente!')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o Assistente!')
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
        # Lista os arquivos da pasta 'arquivos'
        arquivos = [f for f in os.listdir('arquivos') if os.path.isfile(os.path.join('arquivos', f))]
        if not arquivos:
            st.warning("Nenhum arquivo encontrado na pasta 'arquivos'.")
            return  # Sai da função se não houver arquivos

        nome_arquivo = st.selectbox('Selecione o arquivo', arquivos)

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
        if nome_arquivo:
            carrega_modelo(provedor, modelo, api_key, os.path.join('arquivos', nome_arquivo))

    if st.button('️ Limpar o histórico de conversação', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    # ... (resto do código principal)

if __name__=='__main__':
    main()