# Ollama Setup Guide - Local AI with gemma3:1b

This guide explains how to configure the Agentic RAG system to use **Ollama** with the **gemma3:1b** model running completely locally on your machine.

## Why Use Ollama + gemma3:1b?

- **100% Free**: No API costs whatsoever
- **100% Private**: All data stays on your machine
- **No API Keys**: No registration or authentication needed
- **Fast**: gemma3:1b is a small, efficient 1B parameter model optimized for speed
- **Offline**: Works without internet connection (after initial setup)

## Prerequisites

- macOS, Linux, or Windows
- At least 2GB of free RAM
- At least 1GB of free disk space

## Step 1: Install Ollama

### macOS

```bash
# Download and install from the website
# Visit: https://ollama.ai

# OR use Homebrew
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows

Download the installer from [https://ollama.ai/download](https://ollama.ai/download)

## Step 2: Start Ollama Service

Ollama runs as a background service on your machine.

### macOS/Linux

```bash
# Ollama usually starts automatically after installation
# To manually start:
ollama serve
```

This will start the Ollama server at `http://localhost:11434`

### Windows

Ollama starts automatically as a service after installation.

## Step 3: Pull the gemma3:1b Model

```bash
ollama pull gemma3:1b
```

This downloads the gemma3:1b model (approximately 800MB). This only needs to be done once.

**Alternative models** (if you want to try others):

```bash
# Larger, more capable models:
ollama pull llama3.2:3b      # 3B parameters, better quality
ollama pull phi3:mini        # 3.8B parameters, very good

# Even smaller models:
ollama pull phi3:mini-128k   # 3.8B with long context
```

## Step 4: Verify Ollama is Working

Test that Ollama and your model are working:

```bash
ollama run gemma3:1b "Hello, how are you?"
```

You should see a response from the model. Press `Ctrl+D` or type `/bye` to exit.

Check that the Ollama server is running:

```bash
curl http://localhost:11434
```

Should return: `Ollama is running`

## Step 5: Configuration is Already Set!

Your `config.yaml` is already configured to use Ollama:

```yaml
# Embedding Configuration
embeddings:
  provider: "huggingface"
  huggingface_model: "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
llm:
  provider: "ollama"
  ollama_model: "gemma3:1b"
  ollama_base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 1000

# Agent Configuration
agent:
  agent_type: "react"  # Ollama uses ReAct agent
```

## Step 6: Install Python Dependencies

Make sure you have the Ollama Python package:

```bash
pip install -r requirements.txt
```

This includes:
- `ollama` - Ollama Python client
- `sentence-transformers` - For local embeddings

## Step 7: Run the System

Now you can use the RAG system completely locally!

### Ingest Documents

```bash
python ingest_documents.py
```

On first run, it will download the HuggingFace embedding model (sentence-transformers/all-MiniLM-L6-v2, ~90MB). This happens once.

### Run the RAG System

```bash
python main.py
```

Everything runs locally - no API keys, no cloud calls!

## Usage Example

```bash
$ python main.py

============================================================
               AGENTIC RAG SYSTEM
          Knowledge-Based Q&A with AI Agent
============================================================

Using Ollama with model: gemma3:1b
Make sure Ollama is running at: http://localhost:11434
Initializing vector store...
Using local HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2
...

You: What is RAG?

Agent: Thinking...
Agent: Retrieval-Augmented Generation (RAG) is a technique that combines...
```

## Configuration Options

### Change the Model

Edit `config.yaml` to use a different Ollama model:

```yaml
llm:
  ollama_model: "llama3.2:3b"  # Use a different model
```

Make sure to pull the model first:

```bash
ollama pull llama3.2:3b
```

### Change Ollama Server URL

If Ollama is running on a different port or machine:

```yaml
llm:
  ollama_base_url: "http://192.168.1.100:11434"  # Remote Ollama server
```

### Change Embeddings Model

To use a different HuggingFace embedding model:

```yaml
embeddings:
  huggingface_model: "sentence-transformers/all-mpnet-base-v2"  # Better quality, larger
```

## Troubleshooting

### "Connection refused" or "Ollama not reachable"

**Problem**: Can't connect to Ollama server

**Solutions**:
1. Check if Ollama is running:
   ```bash
   curl http://localhost:11434
   ```

2. Start Ollama manually:
   ```bash
   ollama serve
   ```

3. Check if something is using port 11434:
   ```bash
   lsof -i :11434  # macOS/Linux
   ```

### "Model not found"

**Problem**: The gemma3:1b model hasn't been downloaded

**Solution**: Pull the model:
```bash
ollama pull gemma3:1b
```

### Slow Performance

**Problem**: Model responses are slow

**Solutions**:
1. Use a smaller model (gemma3:1b is already small!)
2. Reduce `max_tokens` in config.yaml
3. Check system resources (RAM/CPU usage)
4. Close other applications

### Model Downloads Keep Failing

**Problem**: HuggingFace model download fails

**Solutions**:
1. Check internet connection (first download only)
2. Try a smaller embedding model:
   ```yaml
   embeddings:
     huggingface_model: "sentence-transformers/all-MiniLM-L6-v2"
   ```
3. Clear the cache:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

### Out of Memory Errors

**Problem**: System runs out of RAM

**Solutions**:
1. Use a smaller model:
   ```bash
   ollama pull gemma3:1b  # Smallest option
   ```
2. Close other applications
3. Reduce the number of retrieved chunks:
   ```yaml
   retrieval:
     top_k: 2  # Instead of 4
   ```

## Useful Ollama Commands

```bash
# List installed models
ollama list

# Remove a model
ollama rm gemma3:1b

# Update a model
ollama pull gemma3:1b

# Check Ollama version
ollama --version

# See running models
ollama ps

# Stop a running model
ollama stop gemma3:1b
```

## Performance Tips

1. **Keep Ollama running**: Leave `ollama serve` running in the background for faster responses
2. **Pre-load the model**: Run `ollama run gemma3:1b` once to load it into memory
3. **SSD recommended**: Store models on an SSD for faster loading
4. **RAM matters**: More RAM = better performance (4GB+ recommended)

## Comparing Models

| Model | Size | RAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| gemma3:1b | 800MB | 2GB | Very Fast | Good |
| llama3.2:3b | 2GB | 4GB | Fast | Better |
| phi3:mini | 2.3GB | 4GB | Fast | Excellent |
| mistral:7b | 4GB | 8GB | Medium | Excellent |

## Privacy Benefits

When using Ollama locally:
- ✅ No data sent to external APIs
- ✅ No usage tracking
- ✅ No API key required
- ✅ Works offline (after setup)
- ✅ Complete control over your data
- ✅ No rate limits or quotas

## Cost Comparison

| Provider | Monthly Cost | Privacy | Speed |
|----------|--------------|---------|-------|
| **Ollama (gemma3:1b)** | **$0** | **100% Private** | **Fast** |
| Gemini (free tier) | $0 | Data sent to Google | Fast |
| OpenAI (GPT-4o-mini) | ~$5-20 | Data sent to OpenAI | Fast |

## Next Steps

Once you have Ollama running:

1. Start ingesting documents: `python ingest_documents.py`
2. Run the RAG system: `python main.py`
3. Try different models by editing `config.yaml`
4. Experiment with different embedding models

Enjoy your fully local, private RAG system!
