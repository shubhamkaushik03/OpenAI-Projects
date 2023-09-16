import streamlit as st
import chromadb
import textwrap

# importing openai to get openai interface to allow large language models
from langchain.llms import OpenAI
# import embedding to convert data into vector or mathematical format to increase readbility 
# way to get embeddings from OpenAI's LLMs.
from langchain.embeddings import OpenAIEmbeddings
# importing directory loader to load any document from directory
from langchain.document_loaders import DirectoryLoader
#  provides a way to load PDF documents.
from langchain.document_loaders import PyPDFLoader
# provides a way to split text into characters
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
import openai
openai.api_key = "xyz"
# openai.Model.list()
import os
os.environ['OPENAI_API_KEY'] = openai.api_key

st.header('Pdf reader using openAI')
query_input=st.text_input('Query')
print(query_input)

value = st.button('Submit Query')

def loader(file_path,loader_cls ):
    dataloader = DirectoryLoader(file_path, glob="*.pdf", loader_cls=loader_cls)
    documents= dataloader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)    
    return texts

def search(search_query):
    loader_output= loader('.', PyPDFLoader)
    embedding = OpenAIEmbeddings()
    chroma_client = chromadb.Client()
    vectordb = Chroma.from_documents(documents=loader_output,
                                 embedding=embedding,
                                 client=chroma_client, collection_name="pdf_collection2")
    # query = search_query
    # answer = vectordb.similarity_search(search_query)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    
    return qa_chain(search_query)


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    return (wrap_text_preserve_newlines(llm_response['result']))
    # for source in llm_response["source_documents"]:
    #     return (source.metadata['source'])
    #     # print(source.metadata['source'])

if value:
    llm_response = search(query_input)
    # output=process_llm_response(llm_response)
    output=llm_response['result']
    st.write(output)



