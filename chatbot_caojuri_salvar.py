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
   
hide_streamlit_style = """
            <style>
            /* Oculta o botão do menu principal */
            button[data-testid="stBaseButton-header"] {
                visibility: hidden;
                display: none; /* Garante que não ocupe espaço */
            }
            
            /* Opcional: Oculta a barra de decoração superior (onde o menu fica) */
            /* Descomente a linha abaixo se quiser ocultar toda a barra cinza superior */
            /* div[data-testid="stDecoration"] { visibility: hidden; display: none; } */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image('images/chatbot4.png')

# Defina a API_KEY diretamente aqui
#OPENAI_API_KEY = 'sua-chave-api-aqui'
GOOGLE_API_KEY = st.secrets["google"]["api_key"]


MEMORIA = ConversationBufferMemory()

#pasta_arquivos = os.path.join('mount', 'src','chatbot_caojuri2', 'arquivos')
#pasta_arquivos = '/mount/src/chatbot_caojuri2/arquivos/'

  
def carrega_arquivos(pasta_arquivos):
    import requests
    from urllib.parse import quote
    
    documentos = []
    
    # URLs dos arquivos no GitHub
    base_url = "https://raw.githubusercontent.com/jonhsel/chatbot_caojuri2/master/arquivos/"
    arquivos = ["Apresentação aos Jurados.md",
                "Apresentação - Tribunal do Júri Faculdade.md",
                "ATO-REG 22-2020.md", 
                "ATOREG - 262022 Planejamento atribuição SEPLAG.md",
                "CAOJÚRI - MPMA.md",
                "Código 1ª Fase.md",
                "Código 2ª Fase.md",
                "Contatos institucionais - Procuradores e Promotores.md",
                "Decálogo do Promotor do Júri.md",
                "Informações.md",
                "Júri -  recorte.md",
                "Manual de manual de resolutividade CNMP.md",
                "ODS.md",
                "Projeto Jurado Voluntário.md",
                "Promotorias do Júri.md" ,
                "NTC-CAOJURI22025.md"          
                ]
    
    st.info(f"Tentando carregar {len(arquivos)} arquivos do GitHub...")
    
    for nome_arquivo in arquivos:
        # Codifica o nome do arquivo para URL
        nome_arquivo_encoded = quote(nome_arquivo)
        url = base_url + nome_arquivo_encoded
        
        st.info(f"Tentando carregar: {url}")
        
        try:
            # Faz uma requisição HTTP para obter o conteúdo do arquivo
            response = requests.get(url)
            
            # Verifica se a requisição foi bem-sucedida
            if response.status_code == 200:
                # Obtém o conteúdo do arquivo
                conteudo = response.text
                st.success(f"Arquivo '{nome_arquivo}' carregado com sucesso!")
                documentos.append(conteudo)
            else:
                st.error(f"Erro ao baixar '{nome_arquivo}': Status code {response.status_code}")
                
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {nome_arquivo}: {str(e)}")
    
    if not documentos:
        st.error("Nenhum documento foi carregado com sucesso.")
        return ""
    
    st.success(f"Total de {len(documentos)} documentos carregados com sucesso!")
    return "\n\n".join(documentos)

def carrega_modelo(documentos):
    if not documentos:
        st.error("Nenhum documento foi carregado. Verifique a pasta 'arquivos'.")
        return

    system_message = f''' Você é o chatbot virtual do CAOJÚRI.

    Você possui acesso às seguintes informações vindas dos documentos:

    - 1. Utilize apenas as informações fornecidas nos documentos para basear suas respostas.

    - 2. Sua função é responder questionamentos e fornecer informações sobre temas jurídicos relacionados ao Tribunal do Júri e os temas do conteúdo dos documentos.

    - 3. Você, assistente virtual, foi idealizado, pensado e construído na Coordenação do Promotor de Justiça, Dr. Sandro Carvalho Lobato de Carvalho e na parte
    técnica pelo Assessor Técnico, Jonh Selmo de Souza do Nascimento.

    - 4. Quando forem solicitadas informações sobre o CAOJÚRI, busque as informações apenas do documento: "CAOJÚRI-MPMA.md";

    - 5. Caso seja feito algum questionamento sobre temas não jurídicos relacionado ao Tribunal do Júri, ou sobre os documentos de sua base de dados, se desculpe e peça para reformular o questionamento.

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
    st.header('🤖 PARQUET - Chat Virtual do CAOJÚRI')
    st.write('Minha função é responder questionamentos relacionados ao Tribunal do Júri, estrutura do MPMA, resolutividade do Ministério Público e sobre o direito das vítimas!')

    chain = st.session_state.get('chain')
    if chain is None:
        # Mensagem de erro persiste até que 'chain' seja inicializado
        st.error('⚠️ Iniciar o chatbot na aba lateral!')
        # Reseta o flag para que as mensagens de sucesso/info apareçam na próxima inicialização
        st.session_state.chat_init_messages_shown = False
        st.stop() # Impede a execução do resto da página se o chat não está pronto
    else:
        # Verifica se as mensagens de inicialização já foram mostradas nesta sessão
        if not st.session_state.get('chat_init_messages_shown', False):
            # Usa st.toast para mensagens temporárias (desaparecem após 5s por padrão)
            st.toast('✅ CAOJURICHAT carregado com sucesso!', icon='✅')
            # Pode adicionar um pequeno delay artificial se quiser garantir a ordem dos toasts
            # time.sleep(0.1) # Geralmente não necessário
            st.info('Devido ao alto volume de dados, nosso chat processa até 2 perguntas por minuto. Se ocorrer erro/demora, aguarde 1 min e tente novamente. Agradecemos a compreensão!', icon='ℹ️')

            # Marca que as mensagens foram mostradas para não repetir nesta sessão
            st.session_state.chat_init_messages_shown = True

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
    #tabs_assistente = st.tabs(['Seleção de Arquivos'])
    #with tabs_assistente[0]:
        #st.write("Os arquivos serão carregados diretamente do GitHub.")
        
    # Botão para iniciar o assistente
    if st.button('▶️ Iniciar o Parquet', use_container_width=True):
        try:
            documentos = carrega_arquivos(None)  # Não precisa mais de pasta_arquivos
            
            if documentos:
                carrega_modelo(documentos)
            else:
                st.error("Não foi possível carregar os documentos.")
        except Exception as e:
            st.error(f"Erro ao iniciar o assistente: {str(e)}")
    
    if st.button('️🧹Limpar o histórico de conversação', use_container_width=True):
        st.session_state['memoria'] = MEMORIA
def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__=='__main__':
    main()

