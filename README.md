# Agentic RAG System

A Knowledge-Based Question-Answering system using **Retrieval-Augmented Generation (RAG)** with an **agentic AI architecture**. This project implements a complete data pipeline that enables Large Language Models (LLMs) to answer questions based on custom document collections.

## Overview

This system goes beyond basic RAG by implementing an **agentic approach** where an AI agent:
- Decides when and how to search the knowledge base
- Uses multiple tools to gather information
- Reasons about the best way to answer questions
- Maintains conversation context across interactions

## Features

- **Multi-format Document Support**: PDF, TXT, Markdown, DOCX
- **Intelligent Chunking**: Configurable text splitting with overlap to preserve context
- **Multiple LLM Providers**:
  - **Ollama (default)** - 100% local, private, free
  - Google Gemini - Cloud, free tier available
  - OpenAI - Cloud, paid
- **Vector Embeddings**: Support for HuggingFace (local), Gemini, and OpenAI
- **ChromaDB Integration**: Efficient local vector database with persistence
- **Agentic Architecture**: LangChain-powered agent with custom tools
- **Conversation Memory**: Context-aware multi-turn conversations
- **Interactive CLI**: User-friendly command-line interface
- **Configurable Pipeline**: YAML-based configuration for easy customization
- **Privacy-First**: Default configuration runs 100% locally with no external API calls

## Architecture

```
┌─────────────────┐
│   Documents     │
│  (PDF/TXT/MD)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │
│ Processor       │──► Chunking with overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │
│ Model           │──► OpenAI / HuggingFace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB        │
│ Vector Store    │──► Semantic search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agentic RAG     │
│ - Agent         │
│ - Tools         │──► Knowledge base search
│ - Memory        │    Document statistics
│ - LLM           │    Reasoning & generation
└─────────────────┘
```

## Requirements

- Python 3.9+
- **Ollama with gemma3:1b** (100% free, local, private - **NOW DEFAULT!**)
  - **OR** Google Gemini API key (free tier available)
  - **OR** OpenAI API key (paid)
- See `requirements.txt` for full dependencies

> **Updated!** This system now uses **Ollama** by default for completely local, private, and free AI. No API keys needed! See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for setup instructions.
>
> For cloud options, see [GEMINI_SETUP.md](GEMINI_SETUP.md) (free) or configure OpenAI (paid).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CS5481Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama** (Default - Recommended):
   ```bash
   # Install Ollama (see OLLAMA_SETUP.md for details)
   # macOS: brew install ollama
   # Linux: curl -fsSL https://ollama.ai/install.sh | sh

   # Pull the gemma3:1b model
   ollama pull gemma3:1b

   # No API keys needed! Everything runs locally.
   ```

   **Alternative**: Use cloud providers (requires API keys):
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY or OPENAI_API_KEY
   # Then update config.yaml to change provider
   ```

5. **Add your documents**:
   ```bash
   # Place your documents in the documents/ directory
   cp /path/to/your/documents/*.pdf documents/
   ```

## Usage

### Step 1: Ingest Documents

Process your documents and create the vector database:

```bash
python ingest_documents.py
```

Options:
- `--documents-dir`: Specify custom document directory
- `--config`: Use custom configuration file
- `--reset`: Reset the vector database (delete existing data)

Example:
```bash
python ingest_documents.py --documents-dir ./my_docs --reset
```

### Step 2: Run the RAG System

**Interactive mode** (recommended):
```bash
python main.py
```

**Single question mode**:
```bash
python main.py --question "What is retrieval-augmented generation?"
```

### Interactive Commands

Once in interactive mode, you can use these commands:

- Type your question normally to get an answer
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/stats` - Show knowledge base statistics
- `/quit` - Exit the application

## Configuration

Edit `config.yaml` to customize the system:

### Embedding Models

**OpenAI** (recommended for quality):
```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
```

**HuggingFace** (free, runs locally):
```yaml
embeddings:
  provider: "huggingface"
  huggingface_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### LLM Configuration

**OpenAI**:
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0
  max_tokens: 1000
```

**Ollama** (local, private):
```yaml
llm:
  provider: "ollama"
  ollama_model: "llama2"
```

### Chunking Parameters

Adjust how documents are split:
```yaml
chunking:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks
```

### Retrieval Settings

Configure search behavior:
```yaml
retrieval:
  top_k: 4              # Number of chunks to retrieve
  search_type: "similarity"  # or "mmr" for diversity
```

## Project Structure

```
CS5481Project/
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # Document loading and chunking
│   ├── vector_store.py         # Vector database management
│   ├── agent_tools.py          # Custom tools for the agent
│   └── agent.py                # Agentic RAG implementation
├── documents/                  # Place your documents here
├── chroma_db/                  # Vector database (auto-created)
├── config.yaml                 # Configuration file
├── ingest_documents.py         # Document ingestion script
├── main.py                     # Main application
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## How It Works

### 1. Data Preparation

The `DocumentProcessor` class:
- Loads documents from multiple formats (PDF, TXT, MD, DOCX)
- Splits text into overlapping chunks to preserve context
- Maintains metadata (source, page numbers, etc.)

### 2. Vector Embeddings

The `VectorStoreManager` class:
- Converts text chunks into vector embeddings
- Uses semantic similarity for retrieval
- Persists embeddings in ChromaDB for fast access

### 3. Agentic Layer

The `AgenticRAG` class implements:
- **Agent**: Reasons about how to answer questions
- **Tools**:
  - `knowledge_base_search`: Semantic search over documents
  - `knowledge_base_stats`: Information about available data
- **Memory**: Maintains conversation context
- **LLM**: Generates natural language responses

### 4. Question Answering

When you ask a question:
1. Agent analyzes the question
2. Decides to search the knowledge base
3. Retrieves relevant document chunks
4. Synthesizes an answer based on retrieved context
5. Cites sources when possible

## Example Session

```
You: What is retrieval-augmented generation?

Agent: Retrieval-Augmented Generation (RAG) is a pattern that enables
LLMs to use external data by first retrieving relevant document chunks
from a knowledge base, then using those chunks as context for generating
answers. According to the course notes, this is a core data engineering
task that addresses the limitation that LLMs lack knowledge of specific,
private, or recent documents.

You: How does the chunking work?

Agent: Based on the knowledge base, chunking splits text into digestible
pieces (typically around 1000 characters) with an overlap (e.g., 200
characters). The overlap ensures that context isn't lost if a sentence
is cut in half between chunks. This prevents losing nuance that would
occur if you fed an entire PDF into an embedding model.
```

## Advanced Usage

### Using Local Models (Ollama)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Update `config.yaml`:
   ```yaml
   llm:
     provider: "ollama"
     ollama_model: "llama2"
   ```

### Custom Document Metadata

Add metadata to chunks programmatically:

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.process_documents()

# Add custom metadata
metadata = {"category": "course_notes", "semester": "fall_2024"}
chunks = processor.add_metadata(chunks, metadata)
```

### Programmatic Usage

Use the RAG system in your own code:

```python
from src.agent import AgenticRAG

# Initialize
agent = AgenticRAG("config.yaml")

# Ask questions
response = agent.query("What are the main topics in the knowledge base?")
print(response)

# Clear memory
agent.reset_memory()
```

## Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not found"
Create a `.env` file with your API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### "Vector database not found"
Run the ingestion script first:
```bash
python ingest_documents.py
```

### Out of memory during ingestion
Reduce chunk size or process fewer documents at once:
```yaml
chunking:
  chunk_size: 500  # Smaller chunks
```

## Academic References

1. Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks*. Advances in Neural Information Processing Systems, 33.

2. LangChain Documentation. (2024). *Question Answering over Documents*.

## License

This is a course project for CS5481 - Knowledge Based Systems.

## Contributing

This is an educational project. Feel free to fork and extend for your own learning purposes.
