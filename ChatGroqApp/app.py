import streamlit as st
## docs loader
from langchain_community.document_loaders import PyPDFLoader
## text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
## embeddings
from langchain_huggingface import HuggingFaceEmbeddings
## vectordb
from langchain_community.vectorstores import FAISS
## retriever chain
from langchain.chains import create_retrieval_chain
## documnet chain
from langchain.chains.combine_documents import create_stuff_documents_chain
## groq chat
from langchain_groq import ChatGroq
## chat prompt templete
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

st.title("ChatGroq Q&A Chatboat")


## assign gorq api key
groq_api_key = os.getenv('GROQ_API_KEY')
hf_token = os.getenv("HF_TOKEN")

model = "Llama3-8b-8192"
llm = ChatGroq(groq_api_key=groq_api_key, model=model)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions base on the provided context only.
    Pease provide the most accurate response based on the questions

    <context>
    {context}
    <context>
    Question:{input}
    """
)

## create vector embeddings
def vector_embeddings():
    ## load documents
    loader = PyPDFLoader('/workspaces/HuggingFace/data/Attention.pdf')
    documents = loader.load()
    ## split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents[:50])
    embedding_model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectordb



if st.button("Embed_documents"):
    vectorstore = vector_embeddings()
    st.write("Ready to ask query")
    ## this will ask for user input
    user_prompt = st.text_input("Enter Your query to get response")
    if user_prompt:
        # document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        ## retriever
        retriever = vectorstore.as_retriever()
        ## create ertriever chain
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        response = retriever.invoke({"input": user_prompt})

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------------")




