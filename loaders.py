
from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader,
                                                  CSVLoader,
                                                  PyPDFLoader,
                                                  TextLoader,
                                                  DirectoryLoader,
                                                  UnstructuredMarkdownLoader
)
import os


def carrega_site(url):
    loader = WebBaseLoader(url)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return  documento


def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return  documento 


def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return  documento


def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return  documento


def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return  documento

def carrega_md(caminho):
    try:
        loader = UnstructuredMarkdownLoader(caminho, encoding="utf-8") #encoding UTF-8
        documentos = loader.load()
        documento = "\n\n".join([doc.page_content for doc in documentos])
        return documento
    except Exception as e:
        print(f"Erro ao carregar arquivo {caminho}: {e}") # Usando st.error do Streamlit
        return None  # Retorna None em caso de erro

def carrega_pasta(caminho):
    loader = DirectoryLoader(
        caminho,
        glob="**/*.*",  # Carrega todos os arquivos
        loader_cls={
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }
    )
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento



