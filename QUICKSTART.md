# Quick Start Guide

Get up and running with the Agentic RAG system in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- **Ollama installed** (100% free, local, private - **RECOMMENDED!**)
  - OR Google Gemini API key (cloud, free tier)
  - OR OpenAI API key (cloud, paid)

## Setup Steps

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Set Up Your LLM

**Option A: Using Ollama (Recommended - 100% Free & Private!)**

The system is already configured for Ollama! Just install and pull the model:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the gemma3:1b model
ollama pull gemma3:1b

# No API keys needed! See OLLAMA_SETUP.md for more details.
```

**Option B: Using Google Gemini (Cloud, Free Tier)**

Update `config.yaml`:
```yaml
embeddings:
  provider: "gemini"
  model: "models/embedding-001"

llm:
  provider: "gemini"
  model: "gemini-2.0-flash"

agent:
  agent_type: "react"
```

Then create `.env` with your API key:
```bash
echo "GOOGLE_API_KEY=AIza...your-key-here" > .env
```

**Option C: Using OpenAI (Cloud, Paid)**

Update `config.yaml`:
```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

llm:
  provider: "openai"
  model: "gpt-4o-mini"

agent:
  agent_type: "openai-tools"
```

Then create `.env` with your API key:
```bash
echo "OPENAI_API_KEY=sk-...your-key-here" > .env
```

### 3. Add Documents

The project includes a sample document to get you started:
- `documents/sample_knowledge.md` - Information about RAG systems

Add your own documents:
```bash
cp /path/to/your/documents/*.pdf documents/
cp /path/to/your/documents/*.txt documents/
```

Supported formats: PDF, TXT, MD, DOCX

### 4. Ingest Documents

Process documents and create the vector database:

```bash
python ingest_documents.py
```

You should see output like:
```
Processing documents...
Loaded 1 documents matching *.md
Split 1 documents into 15 chunks
Creating vector embeddings...
Ingestion Complete!
```

### 5. Run the System

Start the interactive chat:

```bash
python main.py
```

## Example Questions

Try asking these questions:

- "What is retrieval-augmented generation?"
- "How does chunking work in RAG systems?"
- "What are the benefits of using RAG?"
- "Tell me about vector embeddings"
- "What is ChromaDB?"
- "What are best practices for RAG systems?"

## Commands

While chatting:
- `/help` - Show available commands
- `/stats` - View knowledge base statistics
- `/clear` - Clear conversation history
- `/quit` - Exit

## Single Question Mode

Ask a question without entering interactive mode:

```bash
python main.py --question "What is RAG?"
```

## Customization

Edit `config.yaml` to customize:

- **Chunk size**: How documents are split
  ```yaml
  chunking:
    chunk_size: 1000
    chunk_overlap: 200
  ```

- **Number of results**: How many chunks to retrieve
  ```yaml
  retrieval:
    top_k: 4
  ```

- **LLM model**: Which model to use
  ```yaml
  llm:
    model: "gemini-2.0-flash"  # or "gemini-1.5-pro" for better quality
  ```

## Troubleshooting

### "Connection refused" when using Ollama
Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434

# Start Ollama if needed
ollama serve
```

### "Model not found" with Ollama
Pull the gemma3:1b model:
```bash
ollama pull gemma3:1b
```

### "GOOGLE_API_KEY not found" or "OPENAI_API_KEY not found"
Only needed if you switched to cloud providers. Make sure you created the `.env` file:
```bash
cat .env
# Should show: GOOGLE_API_KEY=AIza... (for Gemini)
# OR: OPENAI_API_KEY=sk-... (for OpenAI)
```

### "Vector database not found"
Run the ingestion script first:
```bash
python ingest_documents.py
```

### "No documents found"
Add documents to the `documents/` directory and run ingestion again

### Want to start fresh?
Reset the database:
```bash
python ingest_documents.py --reset
```

## Next Steps

1. **Add your own documents**: Replace or add to the sample documents
2. **Experiment with settings**: Try different chunk sizes and retrieval parameters
3. **Explore the code**: Check out the modules in `src/`
4. **Read the full README**: See `README.md` for detailed documentation

## Getting Help

- Check `README.md` for detailed documentation
- Review `config.yaml` for all configuration options
- See the example in `documents/sample_knowledge.md`

Enjoy your Agentic RAG system!
