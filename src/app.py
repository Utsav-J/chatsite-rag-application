# anytime something action happens on the website
# the streamlit server will  reread the code every time
# that causes the global variables to mishbehave and not remain static
# so the st.session_state is used
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv

load_dotenv()


def get_vector_store_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    # chunking down the documents
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create  vector store from the chunks
    vector_store = Chroma.from_documents(documents=document_chunks, 
                                         embedding=OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
                                         )
    
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name='chat_history'),
        ('user',"{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")]
    )

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Answer the following question based on the below context \n\n {context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user','{input}'),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_chain(retriever_chain=retriever_chain)
    response = conversational_rag_chain.invoke({
            'chat_history':st.session_state.chat_history,
            'input':user_query
    })
    return response


#app config
st.set_page_config(page_title="Chatsite", page_icon="😐")
st.title("Chat with websites")



#sidebar
with st.sidebar:
    settings = st.header("Settings")
    website_url = st.text_input("Enter a website")

   

#conditional viewing
if website_url is None  or website_url == "":
    st.info('Please enter a url')
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, How can I help you?"),
        ]
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)

    

    #user input
    user_query = st.chat_input("Type your message here...")
    #chat logic
    if user_query is not None and user_query != "":
        # response = get_response(user_query=user_query) 
        response = get_response(user_query=user_query)
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response['answer']))

        


    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


