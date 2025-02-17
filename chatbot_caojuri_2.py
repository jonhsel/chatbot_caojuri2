import streamlit as st
import tempfile
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI

from loaders import *

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader


#===============
#CSS

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

#################

#st.image('images/juria.png')


TIPOS_ARQUIVOS = ['Arquivos .pdf', 'Site', 'Youtube', 'Arquivos .csv', 'Arquivos .txt']

CONFIG_MODELOS = {  'OpenAI': 
                            {'modelos': ['gpt-4o-mini', 'gpt-4o'],
                            'chat': ChatOpenAI}


}

MEMORIA = ConversationBufferMemory()

def carrega_arquivo(caminho):

    def carrega_pasta():
        documentos = []
        # Certifique-se que a pasta "arquivos" existe
        try:
            for arquivo in os.listdir(caminho):
                if arquivo.endswith(".pdf"): # Verifica se o arquivo é PDF
                    caminho_arquivo = os.path.join(caminho, arquivo)
                    try:
                        loader = PyPDFLoader(caminho_arquivo)
                        documentos.extend(loader.load())
                    except Exception as e:
                        print(f"Erro ao carregar arquivo {arquivo}: {e}")
                else:
                    print(f"Arquivo ignorado por não ser PDF: {arquivo}")
        except FileNotFoundError:
            print(f"Pasta 'arquivos' não encontrada em {os.getcwd()}") # Informa o diretório atual
            return None # Ou [] dependendo do que precisa que sua função retorne
        return documentos
    documents = carrega_pasta()
    return documents


    
    

    # if tipo_arquivo == 'Site':
        
    #     documento = carrega_site(arquivo)
        
        
        

    # if tipo_arquivo == 'Youtube':
    #     documento = carrega_youtube(arquivo)

    # if tipo_arquivo == 'Arquivos .pdf':
    #     with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
    #         temp.write(arquivo.read())
    #         nome_temp = temp.name
    #     documento = carrega_pdf(nome_temp)

    # if tipo_arquivo == 'Arquivos .csv':
    #     with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
    #         temp.write(arquivo.read())
    #         nome_temp = temp.name
    #     documento = carrega_csv(nome_temp)

    # if tipo_arquivo == 'Arquivos .txt':
    #     with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
    #         temp.write(arquivo.read())
    #         nome_temp = temp.name
    #     documento = carrega_txt(nome_temp)
    
    

def carrega_modelo(provedor, modelo, api_key):

    #documento = carrega_arquivo(tipo_arquivo, arquivo)
    caminho = "/arquivos" 
    documento = carrega_arquivo(caminho)
    

    system_message = ''' Você é um assistente técnico chamado 'assistente do Jonh Selmo'.
    Você possui acesso às seguintes informações vindas de um documento:
    """
    {}
    """
    
    Utilize as informações fornecidas para basear suas respostas.

    Sempre que houver $ na saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and coockies to continue", sugira ao usuário carregar novamente o 'Assistente do Jonh Selmo'!
    '''.format(documento)
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
        # chat = st.chat_message(mensagem.type)
        # chat.markdown(mensagem.content)
        with st.chat_message(mensagem.type):  # Use 'with' para o contexto da mensagem
            st.markdown(mensagem.content)
        

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
        
def sidebar():
    #tabs_assistente = st.tabs(['Uploads de Arquivos', 'Modelo de IA'])
    tabs_assistente = st.tabs(['Modelo de IA'])
    #  with tabs_assistente[0]:
    #     tipo_arquivo = st.selectbox('selecione o tipo de URL ou arquivo', TIPOS_ARQUIVOS)
    #     if tipo_arquivo == 'Site':
    #         arquivo = st.text_input('Digite a URL do site')
    #     if tipo_arquivo == 'Youtube':
    #         arquivo = st.text_input('Digite o ID Youtube : Código alfanumérico situado entre "v=" e "&" da URL')
    #     if tipo_arquivo == 'Arquivos .pdf':
    #         arquivo = st.file_uploader('Carregue o arquivo do tipo .pdf', type=['.pdf'])
    #     if tipo_arquivo == 'Arquivos .csv':
    #         arquivo = st.file_uploader('Carregue o arquivo do tipo .csv', type=['.csv'])
    #     if tipo_arquivo == 'Arquivos .txt':
    #         arquivo = st.file_uploader('Carregue o arquivo do tipo .txt', type=['.txt']) 
        
    with tabs_assistente[0]:
        provedor = st.selectbox('Selecione a empresa criadora do modelo de IA', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo de IA', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API do modelo escolhido{provedor}',
            value=st.session_state.get(f'api_key_{provedor}')
        )
        st.session_state[f'api_key_{provedor}'] = api_key    


    if st.button('▶️ Iniciar o Assistente', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key)

    if st.button('🗑️ Limpar o histórico de conversação', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    
    with st.sidebar:
        sidebar()
    pagina_chat()
if __name__=='__main__':
    main()
