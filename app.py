import streamlit as st
import numpy as np
import sqlite3
from pathlib import Path
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain # type: ignore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
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
import pytesseract # type: ignore
from pdf2image import convert_from_path

import uuid
import os
import json
import re

from databases.postgres.postgres import store_prompt_data, store_employee_data, postgres_conn, close_postgres_connection
from databases.sqlite.sqlite import close_sqlite_connection, connect_to_db, clean_sql_query,run_agent_query

from dotenv import load_dotenv
load_dotenv()

# def extract_sqlquery(response_json):
#     try:
#         parsed = json.loads(response_json.replace("'", "\""))  # convert single quotes to double for valid JSON
#         result_text = parsed.get("result", "")

#         # extract the actual SQL query after "SQLQuery:"
#         match = re.search(r"SQLQuery:\s*(.*)", result_text)
#         if match:
#             return match.group(1).strip()
#     except Exception as e:
#         print(f"Error parsing SQL from response: {e}")
    
#     return None

# Set page config
st.set_page_config(page_title="Genesis", page_icon="🤖")

# Load environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["USER_AGENT"] = "GenesisAGI/1.0 (+https://github.com/kenee101/genesis)"
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LOCALDB="USE_LOCALDB"
POSTGRESQL="USE_POSTGRESQL"
db_uri=""

options = ['Use SQLite 3 Database - Students.db', 'Connect to PostgreSQL Database - GENESIS']
selected_option = st.sidebar.radio("Select Database", options)

if options.index(selected_option) == 1:
    db_uri=POSTGRESQL
    postgresql_host = st.sidebar.text_input("PostgreSQL Host", value="localhost")
    postgresql_port = st.sidebar.text_input("PostgreSQL Port", value="5432")
    postgresql_user = st.sidebar.text_input("PostgreSQL User", value="postgres")
    postgresql_password = st.sidebar.text_input("PostgreSQL Password", type="password", value="secret")
    postgresql_db = st.sidebar.text_input("PostgreSQL Database", value="genesis")
    postgres_link = f"postgresql+psycopg2://{postgresql_user}:{postgresql_password}@{postgresql_host}:{postgresql_port}/{postgresql_db}"
else:
    db_uri=LOCALDB

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
    st.title("Welcome to GENESIS")

    if not db_uri:
        st.warning("Please select a database to proceed.")

    sqlite_conn, cursor = connect_to_db('databases/sqlite/students.db')

    #Input Groq API key
    api_key=st.sidebar.text_input("Enter your Groq API key", type="password", value=os.getenv("GROQ_API_KEY"))

    # Initialize Ollama
    model_name = st.sidebar.selectbox(
        "Select Ollama Model",
        ["deepseek-r1", "gemma3", "llava", "codellama"],
        index=0
    )
    
    if api_key:
        llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    # llm = Ollama(model=model_name)
        # session_id=st.text_input("Session ID", value="default_session")
    # if llm:
        @st.cache_resource(ttl=10)
        def configure_db():
            if db_uri == LOCALDB:
                # Use SQLite database
                dbfilepath = (Path(__file__).parent/"databases/sqlite/students.db").absolute()
                print(f"Database file path: {dbfilepath}")
                creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
                return SQLDatabase(engine=create_engine('sqlite:///', creator=creator))
            elif db_uri == POSTGRESQL:
                # Use PostgreSQL database
                if not (postgresql_host and postgresql_port and postgresql_user and postgresql_password and postgresql_db):
                    st.warning("Please enter your PostgreSQL credentials to proceed.")
                    st.stop()
                return SQLDatabase(engine=create_engine(postgres_link))
            
        db = configure_db()

        # Create a SQL database chain
        sql_chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            return_direct=True,
            return_intermediate_steps=True,
            use_query_checker=True,
            top_k=3,
            verbose=True
        )
            
        # Statefully manage chat histories
        if "store" not in st.session_state:
            st.session_state.store = {}

        if "chat_history" not in st.session_state or st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.session_id = '' 
            st.session_state.store = {}

        if "session_id" not in st.session_state:
            st.session_state.session_id = 'default' 
            # str(uuid.uuid4())

        # Use the in-built tool of wikipedia and arxiv
        api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
        api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
        wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
        arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
        search=DuckDuckGoSearchRun(name="DuckDuckGo Search", description="Search the web for relevant information.")


        # User input for questions
        user_question=st.chat_input("Ask any question")
        uploaded_files=st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

        # Process uploaded files
        documents=[]
        # retriever=None
        # webLoader=WebBaseLoader("")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                tempPDF=f"./pdfs/temp.pdf"
                with open(tempPDF, "wb") as file:
                    file.write(uploaded_file.read())
                    file_name=uploaded_file.name

                # Load the PDF
                pdfLoader=PyPDFLoader(tempPDF)
                # Load the documents
                docs=pdfLoader.load()
                # Add the documents to the list
                documents.extend(docs)

        # loaders=[pdfLoader, webLoader]
        # docs=[loader.load() for loader in loaders]
        # Combine the documents
        # docs=[doc for doc_list in docs for doc in doc_list]
        # docs=webLoader.load()
        # documents.extend(docs)

                # Split and create embeddings
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                splits=text_splitter.split_documents(documents)

                if not splits:
                    # Convert PDF pages to images
                    pages = convert_from_path(tempPDF)
                    ocr_texts = []

                    for i, page in enumerate(pages):
                        text = pytesseract.image_to_string(page)
                        if text.strip():
                            ocr_texts.append(Document(page_content=text, metadata={"source": f"{file_name}_page_{i+1}"}))

                    if not ocr_texts:
                        st.error("OCR also failed to extract text from the scanned PDF.")
                        st.stop()

                    splits = text_splitter.split_documents(ocr_texts)

                vectorestore=FAISS.from_documents(documents=splits, embedding=embeddings)
                retriever=vectorestore.as_retriever()
                

                retriever_tool=create_retriever_tool(
                    retriever=retriever,
                    name="pdf_retriever",
                    description="A tool to retrieve relevant documents from the PDF.",
                )

        else:
            retriever_tool=None
            retriever=None


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
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        else:
            history_aware_retriever = None  # or raise a warning
        # history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            """
            You are an intelligent assistant with access to the following tools:

            - SQL Agent: Use ONLY if the user's question is explicitly about structured database information such as student names, IDs, employees. Do NOT use this for general reasoning.
            - Retriever: Use this to answer questions about documents or general facts that may be stored in a knowledge base.
            - LLM Chat: Use this for everything else especially open-ended, general, or conversational questions.

            Always think step by step, and only call a tool when necessary.

            If a user's question can be answered directly from reasoning or general knowledge, use Hybrid RAG.

            If you're unsure which to use, prefer LLM Chat unless the user explicitly mentions data or the database.

            Given the question and the context, provide a lengthy concise answer.

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

        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        if retriever is not None:
            retrieval_chain=create_retrieval_chain(
                retriever=history_aware_retriever,
                combine_docs_chain=qa_chain,
            )
        else:
            retrieval_chain=None


        # Create the conversational RAG chain
        rag_chain = None
        if retrieval_chain is not None:
            rag_chain=RunnableWithMessageHistory(
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

        # Initialize the tools
        tools = [
            Tool(
                name="SQL Agent",
                func=execute_sql_query,
                description="Use this tool ONLY if the user's question explicitly asks for database info like tables, student IDs, employees, or course records."
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
                name="SearchWeb",
                func=search.run,
                description="Retrieves information from the web."
            )
        ]

        if retriever_tool is not None:
            # Add PDF Retriever tool
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
                    func=lambda query: rag_chain.invoke({"input": query}, {'configurable': {'session_id': st.session_state.session_id}}),
                    description="Use this for any general conversation, reasoning, or non-database questions."
                )
            )

        # Initialize the agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,
            early_stopping_method="generate",
            agent_kwargs={
                "prefix": """You are a helpful AI assistant. You have access to the following tools. Use them in this order of priority:
1. PDF Retriever (if available) - Use this FIRST for any questions about uploaded PDFs
2. SQL Agent - Use this for database queries
3. LLM Chat - Use this for general conversation
4. Wikipedia/Arxiv - Use these for factual information
5. Web Search - Use this ONLY as a last resort

When using the PDF Retriever:
- Return the content directly without showing your thought process
- Format the content in a clear, readable way
- If the content is a list, present it as a bulleted list
- If the content has sections, organize it with headers""",
                "suffix": """Begin!

Question: {input}
{agent_scratchpad}""",
                "format_instructions": """Follow this format exactly. Do not include any additional text or notes.

For PDF content:
Question: [user's question]
Thought: I should use the PDF Retriever to find information in the uploaded documents
Action: PDF Retriever
Action Input: [user's question]
Observation: [retrieved information]
Thought: I will format this information clearly for the user
Final Answer: [formatted content from the PDF]

For SQL queries:
Question: [user's question]
Thought: I need to query the database
Action: SQL Agent
Action Input: SELECT [columns] FROM [table] WHERE [conditions] ORDER BY [column] LIMIT [number]
Observation: [query results]
Thought: [interpret results]
Final Answer: [formatted response]

For other questions:
Question: [user's question]
Thought: [your reasoning]
Action: [tool name from {tool_names}]
Action Input: [input for the tool]
Observation: [tool's response]
Thought: [final reasoning]
Final Answer: [your response]

CRITICAL RULES:
1. For SQL queries, ALWAYS use the SQL format above
2. Do not add any text before or after the format
3. Do not add any explanations or notes
4. Do not add any formatting or styling
5. Do not add any tags or markers
6. Do not repeat the format instructions
7. Do not add any greetings or closings
8. Do not use <think> tags or any other formatting tags
9. For SQL queries, ONLY include the actual SQL statement in Action Input - NO markdown formatting, NO code blocks, NO backticks
10. SQL queries must be plain text only - no ```sql or any other formatting"""
            }
        )

        # data = [("fola", 20, "Engineering", 50000),
        #         ("elijah", 18, "Engineering", 60000),
        #         ("Sophia", 20, "Engineering", 70000),
        #         ("Michael", 22, "Engineering", 80000),
        #         ("Sarah", 19, "Engineering", 90000),
        #         ("David", 21, "Engineering", 100000),
        #         ("Emily", 23, "Engineering", 110000)]

        # store_employee_data(data)

        if user_question:
            #  Create prompt embedding
            embedding=embeddings.embed_query(user_question)

            # Get the session history
            st.session_state.chat_history=get_session_history(st.session_state.session_id)
            st.session_state.chat_history.add_user_message(user_question)

            # Run the chain
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response=agent.run({"input": user_question}, callbacks=[st_cb],)
            # response=agent.run({"input": st.session_state.chat_history.messages}, callbacks=[st_cb],)
            
            # memory = ConversationBufferMemory(
            #     memory_key="chat_history",
            #     return_messages=True
            # )
            # # conversation = ConversationChain(llm=llm, memory=memory)
            # qa = ConversationalRetrievalChain.from_llm(
            #     llm=llm,
            #     retriever=retriever,
            #     memory=memory
            # )
            # response = qa.run(user_question)
            
            # sql_query = extract_sqlquery(response)
            # if sql_query:
            #     # Run the SQL query
            # cleaned_query = clean_sql_query(sql_query)
            # result = run_agent_query(cleaned_query, "databases/sqlite/students.db")
            # st.session_state.chat_history.add_ai_message(result)
            # st.success(f"Answer: {result}")
            # store_prompt_data(st.session_state.session_id, user_question, result, embedding, "Gemma2-9b-It")  # embedding must be list

            # else:
            # Display the answer
            with st.chat_message("assistant"):
                st.write(response)
            # st.write_stream(f"Answer: {response}")

            # store_prompt_data(st.session_state.session_id, user_question, response, embedding, "Gemma2-9b-It")  # embedding must be list

            
            # Add the assistant's message to the chat history
            st.session_state.chat_history.add_ai_message(response)
        
            # Store conversation
            # st.session_state.chat_history.append(("user", user_question))
            # st.session_state.chat_history.append(("ai", response))

            # Display the chat history
            # st.write("Chat history:", chat_history.messages)

            # for msg in st.session_state.chat_history.messages:
            #     st.write(f"{msg.type.capitalize()}: {msg.content}")
            
            #  Close the database connection

    else:
        st.warning("Please enter your Groq API key to proceed.")
        close_sqlite_connection(sqlite_conn)
        close_postgres_connection(postgres_conn)

# If no tool is needed:
# Question: [user's question]
# Thought: [your reasoning]
# Action: None
# Action Input: None
# Observation: No tools needed
# Thought: [final reasoning]
# Final Answer: [your response]