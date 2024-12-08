# RAG Model Comparison Tool

An application for comparing different RAG (Retrieval-Augmented Generation) implementations across various LLM providers, helping you determine the optimal setup for your specific use case.

## Quick Start

### First-time Setup
```bash
docker compose up --build
```

### Subsequent Starts
```bash
docker compose up
```

Visit `http://localhost:3000` to access the application.

## Features

- Support for multiple LLM providers:
    - OpenAI (GPT models)
    - Ollama (local models)
    - *Claude (Coming soon)*
- Document processing with automatic image and table extraction
- Real-time chat interface
- Document management system
- Provider-specific model configuration
- Hybrid search combining dense and sparse retrievers

## Important Usage Notes

### Document Processing

**⚠️ Important**: Document embeddings are provider-specific and are generated at upload time based on your current configuration.

- Documents are embedded using all currently configured providers at the time of upload
- If you change an embedding model after uploading documents, you'll need to re-upload them to generate new embeddings
- Documents uploaded when a provider isn't configured won't have embeddings for that provider, even if you configure it later

### Provider Setup

Before uploading documents, ensure all desired providers are properly configured:

1. Configure all providers you want to use first
2. Upload your documents
3. Each configured provider will process the documents with its respective embedding model

If you need to change embedding models or add new providers later, you'll need to:
1. Delete the existing documents
2. Update your configuration
3. Re-upload the documents

## Configuration Guide

### OpenAI Setup
1. Obtain an API key from platform.openai.com
2. Select your preferred model and embedding model
3. Save the configuration

### Ollama Setup
1. Ensure Ollama is running locally (http://ollama:11434 by default)
2. Enter your preferred model name (e.g., llama2, mistral)
3. Specify the embedding model (e.g., nomic-embed-text)
4. Save the configuration

## Use Cases

This tool is designed for:
- Comparing different LLM providers and models
- Testing various embedding models for your specific data
- Evaluating RAG performance with different configurations
- Analyzing how different models handle various document types and queries
- Finding the optimal balance between cost, performance, and accuracy

## Technical Details

The application uses:
- Next.js frontend with Tailwind CSS
- FastAPI backend
- PostgreSQL database
- ChromaDB for vector storage
- Hybrid search combining dense embeddings and BM25
