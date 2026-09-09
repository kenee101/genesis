# Genesis - AI Assistant

A powerful AI assistant built with Streamlit and LangChain.

## Features

- Natural language interaction
- PDF document processing
- Web content analysis
- Database querying
- Academic paper search
- Wikipedia integration
- Code generation and execution
- Enhanced response formatting

## Deployment Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.streamlit/secrets.toml`
4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Add your secrets in the "Secrets" section:

   ```toml
   # API Keys
   HF_TOKEN = "your-huggingface-token"

   # Database Configuration
   POSTGRES_HOST = "your-postgres-host"
   POSTGRES_PORT = "5432"
   POSTGRES_USER = "your-postgres-user"
   POSTGRES_PASSWORD = "your-postgres-password"
   POSTGRES_DB = "your-postgres-db"
   ```

7. Click "Deploy"

## Environment Variables

- `HF_TOKEN`: Your HuggingFace token
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: PostgreSQL database name

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

MIT License

# 🧠 ARTIFICIAL GENERAL INTELLIGENCE - ALL IN ONE

Welcome to the future of AI. This project is an **end-to-end Generative AI stack**, combining the power of **LangChain**, **multi-agent orchestration**, **code understanding**, and advanced **database integrations** — all under one roof.

---

## 🚀 Features

### 🤖 LangChain Tools & Agents

- Build contextual, chain-aware AI applications.
- Utilize LangChain agents for tool-assisted reasoning.

### 🧑‍💻 Multi-Language Code Assistant

- AI that reads, understands, and generates code in multiple languages.
- Smart context switching for dev workflows.

### ✂️ Text Summarization

- Clean, concise summaries powered by LLMs.
- Ideal for document digestion and report generation.

### 💬 Chat App with SQL

- Conversational interface backed by:
  - 📊 **PostgreSQL / MySQL**
  - 📦 **SQLite 3** 

### 🧬 Hybrid Search RAG

- Combine **semantic + keyword** retrieval with RAG (Retrieval Augmented Generation).
- Use **vector databases** like:
  - `FAISS`

---

## 🛠️ Tech Stack

- **LangChain** ⚙️
- **PostgreSQL / MySQL / SQLite** 🗃️
- **FAISS ** 📡
- **Streamlit**
- **Python 3.12+**

---

## 📦 Setup

```bash
git clone https://github.com/kenee101/genesis.git
cd genesis
pip install -r requirements.txt
```
