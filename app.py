import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import Tool
# from login import login_page

import os

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["USER_AGENT"] = "GenesisAGI/1.0 (+https://github.com/kenee101/genesis)"
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "db"

# Check if the user is authenticated

if not st.experimental_user.is_logged_in:
    st.title("Login Page")
    st.header("This app is private.")
    st.write("Please log in to access the application.")
    st.write("You can also use your Groq API key to access the application.")
    # login_page()
    st.button("Log in with Google", on_click=st.login)

else:    
    st.button("Log out", on_click=st.logout)
    st.write(f"Hello, {st.experimental_user.name}!")
    st.title("Conversational RAG with PDF uploads using LangChain and Groq")
    st.write("Upload PDFs and ask questions about their content.")

    #Input Groq API key
    api_key=st.sidebar.text_input("Enter your Groq API key", type="password")

    if api_key:
        llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
        session_id=st.text_input("Session ID", value="default_session")

        # Statefully manage chat histories
        if "store" not in st.session_state:
            st.session_state.store = {}

        # Use the in-built tool of wikipedia and arxiv
        api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
        api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
        wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
        arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
        search=DuckDuckGoSearchRun(name="DuckDuckGo Search", description="Search the web for relevant information.")


        # User input for questions
        user_question=st.text_input("Ask a question about the PDF content")
        uploaded_files=st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

        # Process uploaded files
        if uploaded_files:
            documents=[]
            for uploaded_file in uploaded_files:
                tempPDF=f"./pdfs/temp.pdf"
                with open(tempPDF, "wb") as file:
                    file.write(uploaded_file.read())
                    file_name=uploaded_file.name

                # Load the PDF
                pdfLoader=PyPDFLoader(tempPDF)
                webLoader=WebBaseLoader("https://docs.smith.langchain.com/")
                loaders=[pdfLoader, webLoader]
                # Load the documents
                docs=[loader.load() for loader in loaders]
                # Combine the documents
                docs=[doc for doc_list in docs for doc in doc_list]
                # Add the documents to the list
                documents.extend(docs)

            # Split and create embeddings
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            splits=text_splitter.split_documents(documents)
            vectorestore=FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever=vectorestore.as_retriever()

            retriever_tool=create_retriever_tool(
                retriever=retriever,
                name="pdf_retriever",
                description="A tool to retrieve relevant documents from the PDF.",
            )



            contextualize_q_system_prompt = (
                """
                Given a chat history and the latest user question
                which might reference context in the chat history,
                formulate a standalone question which can be understood
                without the chat history. DO NOT answer the question,
                just reformulate it if needed and otherwise return it as is
                """
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"), 
                ]
            )

            history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Answer question prompt
            system_prompt = (
                """
                You are a helpful assistant for question answering tasks.
                Given the question and the context, provide a concise answer.
                If you don't know the answer, say "I don't know".
                Use the context to support your answer.
                \n\n
                Context: {context} 
                """
            )

            answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            qa_chain= create_stuff_documents_chain(llm,answer_prompt)
            retrieval_chain=create_retrieval_chain(
                retriever=history_aware_retriever,
                combine_docs_chain=qa_chain,
            )

            def get_session_history(session:str)-> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            # Create the conversational RAG chain
            rag_chain=RunnableWithMessageHistory(
                retrieval_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
                verbose=True,
            )

            # Initialize the tools
            tools = [
                Tool(
                    name="PDF Retriever",
                    func=retriever_tool.run,
                    description="Retrieve relevant documents from the PDF."
                ),
                Tool(
                    name="Wikipedia Search",
                    func=wiki_tool.run,
                    description="Search Wikipedia for relevant information."
                ),
                Tool(
                    name="Arxiv Search",
                    func=arxiv_tool.run,
                    description="Search Arxiv for academic papers."
                ),
                Tool(
                    name="HybridRAG",
                    func=rag_chain.invoke,  # or just my_rag_chain if it's a Runnable
                    description="Retrieves information from the hybrid RAG system."
                ),
                Tool(
                    name="SearchWeb",
                    func=search.run,  # or just my_rag_chain if it's a Runnable
                    description="Retrieves information from the web."
                )
            ]

            # Initialize the agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )

            if user_question:
                # Get the session history
                chat_history=get_session_history(session_id)
                chat_history.add_user_message(user_question)

                # Run the chain
                st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                response=agent.run({"input": user_question}, callbacks=[st_cb],)# config={"configurable": {"session_id": session_id}},)

                # Add the assistant's message to the chat history
                chat_history.add_ai_message(response["answer"])

                # Display the answer
                st.success(f"Answer: {response['answer']}")
                st.write("Chat history:", chat_history.messages)
                

    else:
        st.warning("Please enter your Groq API key to proceed.")
