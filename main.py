# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

## 제목
st.title("ChatPDF")
st.write("---")

## OpenAI KEY 입력 받기
openai_key = st.text_input("OPEN_AI_KEY", type="password")

## 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

## 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    # # Load example document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    ## Embedding
    #embeddings_model = OpenAIEmbeddings(openai_key=openai_key)
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
    ## load it into chroma
    db = Chroma.from_documents(texts, embeddings_model)

    ## Question
    st.header("PDF에게 질문해 보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner('Wait for it...'):
            #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_key=openai_key)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            response = qa_chain({"query": question})
            st.write(response["result"])
