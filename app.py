from auth.supabase_auth import sign_up, sign_in, sign_out, get_current_user, reset_password
from enhanced_parser import process_agent_response, display_enhanced_response, get_enhanced_response
import streamlit as st
import numpy as np
import sqlite3
from pathlib import Path
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain  # type: ignore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
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
from sqlalchemy import create_engine
from PIL import Image
import pytesseract  # type: ignore
from pdf2image import convert_from_path

import uuid
import os
import json
import re
from datetime import datetime

# from databases.postgres.postgres import store_prompt_dataq, postgres_conn, close_postgres_connection
# from databases.sqlite.sqlite import close_sqlite_connection, connect_to_db, clean_sql_query
# from databases.sqlite.users import register_user, verify_user, get_all_users

from dotenv import load_dotenv
load_dotenv()


# Set page config
st.set_page_config(page_title="Genesis", page_icon="🤖")

# Load environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(st.secrets)
print("Welcome to Genesis")

hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not hf_token:
    st.error("⚠️ HF_TOKEN is required but not found in secrets or environment variables")
    st.stop()

os.environ["HF_TOKEN"] = hf_token
os.environ["USER_AGENT"] = "GenesisAGI/1.0 (+https://github.com/kenee101/genesis)"

# Database configuration
# if st.secrets.get("postgresql"):
#     POSTGRES_HOST = st.secrets.postgresql.host
#     POSTGRES_PORT = st.secrets.postgresql.port
#     POSTGRES_USER = st.secrets.postgresql.user
#     POSTGRES_PASSWORD = st.secrets.postgresql.password
#     POSTGRES_DB = st.secrets.postgresql.database

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LOCALDB = "USE_LOCALDB"
POSTGRESQL = "USE_POSTGRESQL"
db_uri = ""

postgresql_host = None
postgresql_port = None
postgresql_user = None
postgresql_password = None
postgresql_db = None

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'show_register' not in st.session_state:
    st.session_state.show_register = False
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'password' not in st.session_state:
    st.session_state.password = ""
if 'new_email' not in st.session_state:
    st.session_state.new_email = ""
if 'new_password' not in st.session_state:
    st.session_state.new_password = ""
if 'confirm_password' not in st.session_state:
    st.session_state.confirm_password = ""

# Initialize session state for database selection
if 'db_uri' not in st.session_state:
    st.session_state.db_uri = None

# Initialize session state for database configuration
if 'db_path' not in st.session_state:
    st.session_state.db_path = None


def redirect_to_login():
    """Programmatically redirect to login page"""
    st.session_state.authenticated = False
    st.session_state.show_register = False
    st.session_state.email = ""
    st.session_state.password = ""
    st.session_state.new_email = ""
    st.session_state.new_password = ""
    st.session_state.confirm_password = ""
    st.rerun()


# Simple authentication check
if not st.session_state.authenticated:
    st.header("This app is private.")
    st.write("Please log in to access the application.")

    # Toggle between login and register
    if st.button("Register" if not st.session_state.show_register else "Login"):
        st.session_state.show_register = not st.session_state.show_register
        # st.rerun()

    if st.session_state.show_register:
        # Registration form
        st.subheader("Sign Up")
        with st.form("register_form"):
            st.session_state.new_email = st.text_input(
                "Email", value=st.session_state.new_email)
            st.session_state.new_password = st.text_input(
                "Password", type="password", value=st.session_state.new_password)
            st.session_state.confirm_password = st.text_input(
                "Confirm Password", type="password", value=st.session_state.confirm_password)
            register_submitted = st.form_submit_button("Register")

            if register_submitted:
                if st.session_state.new_password != st.session_state.confirm_password:
                    st.error("Passwords do not match!")
                elif len(st.session_state.new_password) < 6:
                    st.error("Password must be at least 6 characters long!")
                else:
                    success, message = sign_up(
                        st.session_state.new_email, st.session_state.new_password)
                    if success:
                        st.success(message)
                        st.session_state.show_register = False
                        # Clear registration fields
                        st.session_state.new_email = ""
                        st.session_state.new_password = ""
                        st.session_state.confirm_password = ""
                        st.rerun()
                    else:
                        st.error(message)
    else:
        # Login form
        st.subheader("Login")
        with st.form("login_form"):
            st.session_state.email = st.text_input(
                "Email", value=st.session_state.email)
            st.session_state.password = st.text_input(
                "Password", type="password", value=st.session_state.password)
            login_submitted = st.form_submit_button("Login")

            if login_submitted:
                success, message = sign_in(
                    st.session_state.email, st.session_state.password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.email = st.session_state.email
                    # Clear login fields
                    st.session_state.password = ""
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # Add password reset option
        if st.button("Forgot Password?"):
            reset_email = st.text_input("Enter your email")
            if st.button("Send Reset Link"):
                success, message = reset_password(reset_email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
else:
    if st.sidebar.button("Log out"):
        sign_out()
        redirect_to_login()

    # Get current user info
    user = get_current_user()
    if user:
        st.title(f"Welcome to GENESIS, {user.email}!")
    else:
        st.title("Welcome to GENESIS!")

    # Add Database Import section
    st.sidebar.markdown("### Please select a database at a time")
    sqlite_db = st.sidebar.checkbox("Import SQLite Database")
    uploaded_db = None
    if sqlite_db:
        uploaded_db = st.file_uploader("Upload SQLite Database", type=[
                                       'db', 'sqlite', 'sqlite3'])

        if uploaded_db is not None:
            try:
                # Create the database directory if it doesn't exist
                home_dir = os.path.expanduser("~")
                db_dir = os.path.join(home_dir, ".genesis", "databases")
                os.makedirs(db_dir, exist_ok=True)

                # Save the uploaded file
                db_path = os.path.join(db_dir, "imported_users.db")
                with open(db_path, "wb") as f:
                    f.write(uploaded_db.getvalue())

                # Verify the database is valid
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                cursor = conn.cursor()

                # Get list of tables
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if tables:
                    st.success(
                        f"Successfully imported database with {len(tables)} tables!")
                    st.write("Tables found:")
                    for table in tables:
                        st.write(f"- {table[0]}")

                    # Update the database path in session state
                    # if st.button("Use This Database"):
                    st.session_state.db_path = db_path
                    st.success(
                        "Database path updated! The new database will be used.")
                    st.rerun()
                else:
                    st.warning("The imported database contains no tables.")

                conn.close()

            except sqlite3.Error as e:
                st.error(f"Error importing database: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    # selected_option = st.sidebar.radio("Select Database", options)
    postgresql_db = st.sidebar.checkbox("Connect to PostgreSQL Database")

    if postgresql_db:
        st.session_state.db_uri = POSTGRESQL
        postgresql_host = st.sidebar.text_input("PostgreSQL Host")
        postgresql_port = st.sidebar.text_input("PostgreSQL Port")
        postgresql_user = st.sidebar.text_input("PostgreSQL User")
        postgresql_password = st.sidebar.text_input(
            "PostgreSQL Password", type="password")
        postgresql_db = st.sidebar.text_input("PostgreSQL Database")
        postgres_link = f"postgresql+psycopg2://{postgresql_user}:{postgresql_password}@{postgresql_host}:{postgresql_port}/{postgresql_db}"
    elif sqlite_db:
        st.session_state.db_uri = LOCALDB
    else:
        st.session_state.db_uri = None

    # sqlite_conn, cursor = connect_to_db('databases/sqlite/students.db')

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("⚠️ Api key is required but not found in secrets or environment variables")
        st.stop()
    # api_key=st.sidebar.text_input("Enter your Groq API key", type="password", value=os.getenv("GROQ_API_KEY"))

    # LLM Selection
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["Groq"],
        index=0
    )

    # Initialize LLM based on selection
    @st.cache_resource(ttl=10)
    def initialize_llm(provider: str, api_key: str = None, model_name: str = None):
        if provider == "Groq":
            if not api_key:
                st.warning(
                    "Please enter your Groq API key to use Groq models.")
                return None
            return ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
        # elif provider == "Ollama":
        #     if not model_name:
        #         st.warning("Please select an Ollama model.")
        #         return None
        #     return Ollama(model=model_name)
        return None

    # Initialize Ollama model selection if Ollama is selected
    # ollama_model = None
    # if llm_provider == "Ollama":
    #     ollama_model = st.sidebar.selectbox(
    #         "Select Ollama Model",
    #         ["deepseek-r1", "gemma3", "codellama"],
    #         index=0
    #     )

    # Initialize the LLM
    llm = initialize_llm(llm_provider, api_key)

    if llm:
        # Initialize SQL chain as None by default
        sql_chain = None

        # Only configure database if user wants to use SQL features
        if (uploaded_db is not None) or (postgresql_port is not None and postgresql_host is not None):
            if st.sidebar.checkbox("Enable SQL Database Features"):
                @st.cache_resource(ttl=10)
                def configure_db():
                    if st.session_state.db_uri == LOCALDB:
                        # Use SQLite database with proper path handling
                        dbfilepath = st.session_state.db_path or st.secrets.get(
                            "sqlite_path")
                        if not dbfilepath:
                            st.error(
                                "No database found. Please import a database.")
                            return None

                        print(f"Database file path: {dbfilepath}")
                        def creator(): return sqlite3.connect(
                            f"file:{dbfilepath}?mode=ro", uri=True)
                        return SQLDatabase(engine=create_engine('sqlite:///', creator=creator))
                    elif st.session_state.db_uri == POSTGRESQL:
                        # Use PostgreSQL database
                        if not (postgresql_host and postgresql_port and postgresql_user and postgresql_password and postgresql_db):
                            st.warning(
                                "Please enter your PostgreSQL credentials to proceed.")
                            return None
                        return SQLDatabase(engine=create_engine(postgres_link))
                    return None

                db = configure_db()
                if db is not None:
                    # Create a SQL database chain only if database is configured
                    sql_chain = SQLDatabaseChain.from_llm(
                        llm=llm,
                        db=db,
                        return_direct=True,
                        return_intermediate_steps=True,
                        use_query_checker=True,
                        top_k=3,
                        verbose=True
                    )

        # Add chat session management
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
            st.session_state.chat_sessions['default'] = {
                "name": "Default Chat",
                "history": ChatMessageHistory(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.session_id = 'default'

        # Add new chat session button
        if st.sidebar.button("➕ New Chat"):
            new_session_id = f"chat_{len(st.session_state.chat_sessions)}"
            st.session_state.chat_sessions[new_session_id] = {
                "name": f"Chat {len(st.session_state.chat_sessions)}",
                "history": ChatMessageHistory(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.session_id = new_session_id
            st.rerun()

        # Display chat sessions in sidebar
        st.sidebar.markdown("### Chat Sessions")
        for session_id, session_data in st.session_state.chat_sessions.items():
            # sessions_items = list(st.session_state.chat_sessions.items())
            # for session_id, session_data in sessions_items:

            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(
                    f"💬 {session_data['name']}",
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    st.session_state.session_id = session_id
            with col2:
                if session_id != 'default':
                    if st.button("🗑️", key=f"delete_{session_id}"):
                        del st.session_state.chat_sessions[session_id]
                        if st.session_state.session_id == session_id:
                            st.session_state.session_id = 'default'
                        st.rerun()

        def get_session_history(session: str) -> BaseChatMessageHistory:
            return st.session_state.chat_sessions[session]["history"]

        # Update the chat history management
        def update_chat_history(message_type: str, content: str):
            current_session = st.session_state.session_id
            if message_type == "user":
                st.session_state.chat_sessions[current_session]["history"].add_user_message(
                    content)
            else:
                st.session_state.chat_sessions[current_session]["history"].add_ai_message(
                    content)

        # Use the in-built tool of wikipedia and arxiv
        api_wrapper_wiki = WikipediaAPIWrapper(
            top_k_results=1, doc_content_chars_max=250)
        api_wrapper_arxiv = ArxivAPIWrapper(
            top_k_results=1, doc_content_chars_max=250)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
        arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
        search_tool = DuckDuckGoSearchRun(
            name="DuckDuckGo Search", description="Search the web for relevant information.")

        # Add web loader tool
        def load_web_content(url: str) -> str:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                if docs:
                    # Return first 1000 chars of content
                    return docs[0].page_content[:1000]
                return "No content found on the webpage."
            except Exception as e:
                return f"Error loading webpage: {str(e)}"

        # User input for questions
        user_question = st.chat_input("Ask any question")
        uploaded_files = st.file_uploader(
            "Choose a PDF file", type="pdf", accept_multiple_files=True)

        # Process uploaded files
        documents = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                tempPDF = f"./pdfs/temp.pdf"
                with open(tempPDF, "wb") as file:
                    file.write(uploaded_file.read())
                    file_name = uploaded_file.name

                # Load the PDF
                pdfLoader = PyPDFLoader(tempPDF)
                # Load the documents
                docs = pdfLoader.load()
                # Add the documents to the list
                documents.extend(docs)

                # Split and create embeddings
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)

                if not splits:
                    # Convert PDF pages to images
                    pages = convert_from_path(tempPDF)
                    ocr_texts = []

                    for i, page in enumerate(pages):
                        text = pytesseract.image_to_string(page)
                        if text.strip():
                            ocr_texts.append(Document(page_content=text, metadata={
                                             "source": f"{file_name}_page_{i+1}"}))

                    if not ocr_texts:
                        st.error(
                            "OCR also failed to extract text from the scanned PDF.")
                        st.stop()

                    splits = text_splitter.split_documents(ocr_texts)

                vectorestore = FAISS.from_documents(
                    documents=splits, embedding=embeddings)
                retriever = vectorestore.as_retriever()

                retriever_tool = create_retriever_tool(
                    retriever=retriever,
                    name="pdf_retriever",
                    description="A tool to retrieve relevant documents from the PDF.",
                )

        else:
            retriever_tool = None
            retriever = None

        contextualize_q_system_prompt = (
            """
            You are an AI assistant.
            Given a chat history and the user questions
            which might reference context in the chat history,
            formulate a standalone question which can be understood
            without the chat history. DO NOT answer the question,
            just reformulate it if needed and otherwise return it as is.
            """
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        if retriever is not None:
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt)
        else:
            history_aware_retriever = None

        system_prompt = (
            """
            You are Genesis, an intelligent AI assistant. When users address you with "hey genesis" or similar greetings, 
            respond in a friendly and helpful manner. You have access to the following tools:

            - SQL Agent: Use ONLY if the user's question is explicitly about structured database information such as student names, IDs, employees. Do NOT use this for general reasoning.
            - Web Loader: Use this to extract and analyze content from specific web pages. Input should be a valid URL. Use this when users ask about specific web pages or want to analyze web content.
            - Wikipedia Search: Use this for general knowledge and factual information from Wikipedia.
            - Arxiv Search: Use this for academic papers and research information.
            - DuckDuckGo Search: Use this for general web searches when other tools don't have the information.
            - PDF Retriever: Use this to answer questions about uploaded PDF documents.
            - LLM Chat: Use this for everything else especially open-ended, general, or conversational questions.

            Tool Usage Guidelines:
            1. SQL Agent: Only for database queries and structured data
            2. Web Loader: For specific webpage content analysis
            3. Wikipedia/Arxiv: For factual and academic information
            4. DuckDuckGo: For general web searches
            5. PDF Retriever: For uploaded document analysis
            6. LLM Chat: For general conversation and reasoning

            Always think step by step, and only call a tool when necessary.

            If a user's question can be answered directly from reasoning or general knowledge, use Hybrid RAG.

            If you're unsure which to use, prefer LLM Chat unless the user explicitly mentions:
            - Database information (use SQL Agent)
            - Specific web pages (use Web Loader)
            - Academic papers (use Arxiv)
            - Wikipedia information (use Wikipedia Search)
            - General web information (use DuckDuckGo)
            - PDF content (use PDF Retriever)

            Provide comprehensive, detailed responses that include:
            1. A clear and thorough explanation of the main concept or answer
            2. Multiple examples or use cases where applicable
            3. Step-by-step breakdowns for complex topics
            4. Related concepts and their connections
            5. Practical applications and real-world scenarios
            6. Potential implications or considerations
            7. Supporting evidence or references when available
            8. Visual or structural organization (lists, sections, etc.) for better readability

            If you don't know the answer, say "I don't know" and explain what information you would need to provide a better response.
            Use the context to support your answer and provide additional relevant information.
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

        qa_chain = create_stuff_documents_chain(llm, answer_prompt)

        if retriever is not None:
            retrieval_chain = create_retrieval_chain(
                retriever=history_aware_retriever,
                combine_docs_chain=qa_chain,
            )
        else:
            retrieval_chain = None

        # Create the conversational RAG chain
        rag_chain = None
        if retrieval_chain is not None:
            rag_chain = RunnableWithMessageHistory(
                retrieval_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
                verbose=True,
            )

        def clean_sql_query(query):
            # Remove any markdown formatting and standardize case
            query = query.replace("```sql", "").replace("```", "").strip()
            # Convert name to lowercase for case-insensitive matching
            if "WHERE name =" in query:
                query = query.replace("WHERE name =", "WHERE LOWER(name) =")
            return query

        def execute_sql_query(query):
            # Clean the query and execute it
            cleaned_query = clean_sql_query(query)
            try:
                result = sql_chain.invoke({"query": cleaned_query})
                # Extract just the result from the response
                if result and "result" in result:
                    return result["result"]
                return "No results found"
            except Exception as e:
                return f"Error executing query: {str(e)}"

        # Initialize the tools with conversation awareness
        tools = [
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
                name="SearchWeb",
                func=search_tool.run,
                description="Retrieves information from the web."
            ),
            Tool(
                name="Web Loader",
                func=load_web_content,
                description="Load and extract content from a webpage URL. Input should be a valid URL."
            )
        ]

        # Add SQL Agent only if sql_chain is available
        if sql_chain is not None:
            tools.insert(0, Tool(
                name="SQL Agent",
                func=execute_sql_query,
                description="Use this tool ONLY if the user's question explicitly asks for database info like tables, students, or course records."
            ))

        # Add PDF Retriever tool if available
        if retriever_tool is not None:
            tools.append(
                Tool(
                    name="PDF Retriever",
                    func=retriever_tool.run,
                    description="Retrieve relevant document information from the PDF."
                )
            )
            # Add LLM Chat tool
            tools.append(
                Tool(
                    name="LLM Chat",
                    func=lambda query: rag_chain.invoke(
                        {"input": query,
                            "chat_history": st.session_state.chat_sessions[st.session_state.session_id]["history"].messages},
                        {'configurable': {'session_id': st.session_state.session_id}}
                    ),
                    description="Use this for any general conversation, reasoning, or non-database questions."
                )
            )

        # Initialize the agent with conversation awareness
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate",
            agent_kwargs={
                "prefix": """You are a helpful AI assistant. Use tools in this order:
1. Web Loader - For specific webpage content analysis
2. PDF Retriever - For PDF questions
3. SQL Agent - For database queries
4. Wikipedia/Arxiv - For facts and academic information
5. DuckDuckGo - For general web searches
6. LLM Chat - For general questions

When using the Web Loader:
- Input must be a valid URL
- Use for direct webpage content analysis
- Extract and summarize webpage content
- Handle errors gracefully if webpage is inaccessible

Format responses with markdown when helpful. Keep answers clear and concise. For general knowledge questions, use LLM Chat directly. For code, use triple backticks with language specification. For images, use clear image generation prompts.""",
                "suffix": """Begin!

Question: {input}
Chat History: {chat_history}
{agent_scratchpad}""",
                "format_instructions": """Format:
Question: [question]
Thought: [reasoning]
Action: [tool]
Action Input: [input]
Observation: [result]
Final Answer: [response]

Rules:
1. Use markdown in Final Answer
2. Keep responses focused
3. Format SQL as plain text
4. Use lists/tables when helpful
5. For general knowledge, use LLM Chat directly
6. Do not display the question, action, action input, observation, thought in the final answer
7. Format code with triple backticks and language spec (e.g. ```python, ```javascript)
8. Do not include any meta-commentary or thought process in Final Answer
9. For code examples, use proper language tags
10. For image requests, use clear prompts
11. For webpage analysis, use Web Loader with valid URLs"""
            }
        )


        st.markdown("""
        <style>
            .stChatMessage {
                border-radius: 1rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: row;
            }
            .stTextInput {
                cursor: pointer !important;
            }
            .stTextInput input {
                cursor: text !important;
            }
            .stChatMessage[data-testid="stChatMessage"] {
                background-color: transparent;
            }
            .user-message {
                background-color: gray;
                border-radius: 15px 15px 0 15px;
                padding: 10px;
                margin: 10px;
                margin-top: -5px;
                display: flex;
                justify-content: flex-end;
                align-items: center;
                gap: 1rem;
            }
            .assistant-message {
                # background-color: #f5f5f5;
                border-radius: 15px 15px 15px 0;
                padding: 10px;
                margin: 10px;
                margin-top: -5px;
                display: flex;
                justify-content: flex-start;
                align-items: center;
                gap: 1rem;
            }
            .message-timestamp {
                font-size: 0.7rem;
                color: black;
                font-weight: bold;
                font-size: 1rem;
                white-space: nowrap;
                flex-shrink: 0;
            }
            .user-message-content, .assistant-message-content {
                # flex: 1;
                min-width: 0;
                overflow-wrap: break-word;
                word-wrap: break-word;
                word-break: break-word;
                white-space: pre-wrap;
            }
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 5px;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 15px;
                margin: 5px 0;
            }
            .typing-dot {
                width: 8px;
                height: 8px;
                background-color: #666;
                border-radius: 50%;
                animation: typing 1s infinite ease-in-out;
            }
            @keyframes typing {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
        </style>
        """, unsafe_allow_html=True)

        # Add typing indicator component
        # def show_typing_indicator():
        #     with st.chat_message("assistant", avatar="🤖"):
        #         st.markdown("""
        #         <div class="typing-indicator">
        #             <div class="typing-dot"></div>
        #             <div class="typing-dot"></div>
        #             <div class="typing-dot"></div>
        #         </div>
        #         """, unsafe_allow_html=True)

        # Display chat history
        if "chat_sessions" in st.session_state:
            current_session = st.session_state.session_id
            current_history = st.session_state.chat_sessions[current_session]["history"]

            if hasattr(current_history, 'messages'):
                for message in current_history.messages:
                    if message.type == "human":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(f"""
                            <div class="user-message">
                                <div class="user-message-content">{message.content}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            if message.content.startswith("__ENHANCED_RESPONSE__"):
                                response_key = message.content.split(
                                    "__ENHANCED_RESPONSE__")[1]
                                stored_response = get_enhanced_response(
                                    response_key)
                                if stored_response != "Response not found":
                                    display_enhanced_response(
                                        stored_response, include_images=True, message_id=response_key)
                            else:
                                st.markdown(f"""
                                <div class="assistant-message">
                                    <div class="assistant-message-content">{message.content}</div>
                                </div>
                                """, unsafe_allow_html=True)

        if user_question:
            try:
                # Create prompt embedding
                embedding = embeddings.embed_query(user_question)

                # Get the session history
                current_history = st.session_state.chat_sessions[st.session_state.session_id]["history"]

                # Display user message
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="user-message-content">{user_question}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Add user message to history
                update_chat_history("user", user_question)

                # Process response
                with st.spinner('Thinking...'):
                    try:
                        st_cb = StreamlitCallbackHandler(
                            st.empty(), expand_new_thoughts=False)
                        response = agent.invoke(
                            {
                                "input": user_question,
                                "chat_history": current_history.messages
                            },
                            callbacks=[st_cb],
                            config={"configurable": {
                                "session_id": st.session_state.session_id}}
                        )

                        # Process the response
                        response_output = response.get(
                            "output", "I apologize, but I couldn't generate a proper response.")

                        # Check if this is a SQL query response
                        if isinstance(response_output, dict) and "result" in response_output:
                            processed_response = response_output["result"]
                        else:
                            processed_response = process_agent_response(
                                response_output, user_question)

                        # Display AI message on the right with timestamp
                        with st.chat_message("assistant", avatar="🤖"):
                            display_text = processed_response
                            if "__ENHANCED_RESPONSE__" in processed_response:
                                display_text = ""

                            st.markdown(f"""
                            <div class="assistant-message">
                                <div class="assistant-message-content">{display_text}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Add the assistant's message to history
                        if processed_response:
                            update_chat_history("ai", processed_response)

                    except Exception as e:
                        error_message = f"An error occurred while processing your request: {str(e)}"

                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(f"""
                            <div class="assistant-message">
                                <div class="assistant-message-content">{error_message}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        if st.button("🔄 Retry", key="retry_button"):
                            st.rerun()

                        update_chat_history("ai", error_message)

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                if st.button("🔄 Refresh", key="refresh_button"):
                    st.rerun()
                st.stop()

    else:
        st.warning("Please enter your Groq API key to proceed.")
        # close_sqlite_connection(sqlite_conn)
        # close_postgres_connection(postgres_conn)
