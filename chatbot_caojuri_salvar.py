import streamlit as st
import tempfile
import os
from datetime import datetime

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import nltk
from loaders import carrega_pdf, carrega_csv, carrega_txt, carrega_md

try:
    nltk.data.find('tokenizers/punkt/punkt.pkl')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

# CSS
with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

st.image('images/chatbot4.png')

# Defina a API_KEY diretamente aqui
#OPENAI_API_KEY = 'sua-chave-api-aqui'
GOOGLE_API_KEY = st.secrets["google"]["api_key"]


MEMORIA = ConversationBufferMemory()

#pasta_arquivos = os.path.join('mount', 'src','chatbot_caojuri2', 'arquivos')
#pasta_arquivos = '/mount/src/chatbot_caojuri2/arquivos/'

  
def carrega_arquivos(pasta_arquivos):
    
    base_url = "https://raw.githubusercontent.com/jonhsel/chatbot_caojuri2/refs/heads/master/arquivos/"
    arquivos = ["ATO-REG 22-2020.md", "CAOJÚRI - MPMA.md"]
    documentos = []    
    
    #for nome_arquivo in os.listdir(pasta_arquivos):
    for nome_arquivo in arquivos:
        caminho_arquivo = base_url + nome_arquivo
        if os.path.isfile(caminho_arquivo):
            try:
                documento = carrega_md(caminho_arquivo)
                
                if documento:
                    documentos.append(documento)

            except Exception as e:
                st.error(f"Erro ao carregar arquivo {nome_arquivo}: {e}")

    return "\n\n".join(documentos)

def carrega_modelo(documentos):
    if not documentos:
        st.error("Nenhum 3 documento foi carregado. Verifique a pasta 'arquivos'.")
        return

    system_message = f''' Você é o chatbot virtual do CAOJÚRI.

    Você possui acesso às seguintes informações vindas de um ou mais documentos:

    - 1. Sua função é responder questionamentos e fornecer informações sobre temas jurídicos relacionados ao Tribunal do Júri.

    - 2. Você, assistente virtual, foi idealizado, pensado e construído na Coordenação do Promotor de Justiça, Dr. Sandro Carvalho Lobato de Carvalho e na parte
    técnica pelo Assessor Técnico, Jonh Selmo de Souza do Nascimento.

    - 3. Utilize apenas as informações fornecidas nos documentos para basear suas respostas.

    - 4. Quando forem solicitadas informações sobre o CAOJÚRI, busque as informações apenas do documento: "CAOJÚRI-MPMA.md";

    - 5. Caso seja feito algum questionamento sobre temas que não seja jurídico relacionado ao Tribunal do Júri, se desculpe e peça para reformular o questionamento.

    - 6. Caso algum link de url seja solicitado, informe apenas os existentes nas base de dados.

    ####
    {documentos}
    ####
    '''

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    #chat = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY)
    chat = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-lite', 
        api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,  # Importante para compatibilidade
        temperature=0.5  # Ajuste a criatividade conforme necessário
    )
    chain = template | chat
    st.session_state['chain'] = chain

def salvar_conversa():
    if 'memoria' in st.session_state:
        memoria = st.session_state['memoria']
        
        # Cria o nome do arquivo com data e hora atual
        nome_arquivo = f"conversa_caojuri_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Prepara o conteúdo do arquivo
        conteudo = ""
        for mensagem in memoria.buffer_as_messages:
            conteudo += f"{mensagem.type.upper()}: {mensagem.content}\n\n"
        
        # Cria uma coluna para o botão de download
        col1, col2 = st.columns([1, 2])
        
        with col1:
            download_button = st.download_button(
                label="📥 Baixar conversa",
                data=conteudo,
                file_name=nome_arquivo,
                mime="text/plain",
                key="download_conversa"
            )
        
        with col2:
            st.info("O arquivo será salvo na pasta de downloads do seu navegador. Você pode movê-lo para outra pasta depois.")
            
        if download_button:
            st.success(f"Conversa '{nome_arquivo}' baixada com sucesso!")
    else:
        st.warning("Não há conversa para salvar.")

def pagina_chat():
    st.header('⚖️ Chat Virtual do CAOJÚRI')
    st.write('Minha função é responder questionamentos relacionados ao Tribunal do Júri!')

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('⚠️ Carregue o arquivo antes de inicializar o assistente!')
        st.stop()

    # Botão para salvar conversa
    if st.button('💾 Salvar Conversa'):
        salvar_conversa()

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
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs_assistente = st.tabs(['Seleção de Arquivos'])
    with tabs_assistente[0]:
        # No Streamlit Cloud, o caminho correto provavelmente é este:
        pasta_arquivos = '/mount/src/chatbot_caojuri2/arquivos'
        
        # Verifique se esse caminho existe
        if not os.path.exists(pasta_arquivos):
            # Tente outros caminhos possíveis
            possiveis_caminhos = [
                os.path.join(os.getcwd(), 'arquivos'),  # Caminho baseado no diretório atual
                'arquivos',  # Caminho relativo simples
                './arquivos',  # Caminho relativo explícito
                '../arquivos',  # Um nível acima
                '/app/arquivos'  # Outro possível caminho no Streamlit Cloud
            ]
            
            # Tente cada caminho até encontrar um que exista
            for caminho in possiveis_caminhos:
                if os.path.exists(caminho):
                    pasta_arquivos = caminho
                    st.info(f"Pasta encontrada em: {pasta_arquivos}")
                    break
            else:  # Se nenhum caminho existir
                st.error("Não foi possível encontrar a pasta 'arquivos'")
                st.write(f"Diretório atual: {os.getcwd()}")
                st.write("Conteúdo do diretório atual:")
                st.write(os.listdir('.'))
                
                # Verifique o diretório raiz do Streamlit Cloud
                if os.path.exists('/mount/src'):
                    st.write("Conteúdo de /mount/src:")
                    st.write(os.listdir('/mount/src'))
                return
        
        # Verificar se a pasta existe e contém arquivos
        if not os.path.exists(pasta_arquivos):
            st.warning(f"A pasta '{pasta_arquivos}' não foi encontrada.")
            return
        
        if not os.listdir(pasta_arquivos):
            st.warning(f"A pasta '{pasta_arquivos}' está vazia.")
            return
        
        # Listar os arquivos encontrados
        st.write(f"Arquivos encontrados na pasta '{pasta_arquivos}':")
        for nome_arquivo in os.listdir(pasta_arquivos):
            st.write(f"- {nome_arquivo}")
        
        # Botão para iniciar o assistente
        if st.button('▶️ Iniciar o Assistente', use_container_width=True):
            try:
                documentos = carrega_arquivos(pasta_arquivos)
                
                # Verificar se a função retornou algum documento
                if not documentos:
                    st.error("Nenhum documento foi carregado, apesar dos arquivos serem encontrados.")
                    return
                
                carrega_modelo(documentos)
            except Exception as e:
                st.error(f"Erro ao carregar arquivos: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        if st.button('️ Limpar o histórico de conversação', use_container_width=True):
            st.session_state['memoria'] = MEMORIA
def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__=='__main__':
    main()

