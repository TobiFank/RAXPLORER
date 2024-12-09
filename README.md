[![License: Custom Non-Commercial](https://img.shields.io/badge/License-Custom%20Non--Commercial-red.svg)](LICENSE.md)

# Advanced RAG Implementation & Analysis Platform

A sophisticated platform for developing and analyzing RAG (Retrieval-Augmented Generation) implementations. Built with a focus on multi-modal document understanding, this tool combines advanced document processing, intelligent retrieval strategies, and comparative analysis across different LLM providers.

![Diagram](images/system_overview.svg)

## Key Features

### Intelligent Document Processing
- Multi-modal processing combining text, tables, and images
- Spatial analysis for layout understanding
- Automatic image-text correlation and caption detection
- Hierarchical document segmentation with context preservation

### Advanced Retrieval System
- Hybrid search combining dense embeddings and BM25
- Multi-stage query decomposition
- Step-back prompting for broader context
- Chat history-aware query rephrasing
- Intelligent result reranking using reciprocal rank fusion

### Flexible Provider Architecture
- Support for OpenAI and Ollama (Claude coming soon)
- Provider-specific embedding optimization
- Real-time performance analysis
- Comparative response evaluation

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

## Technical Architecture

### RAG Pipeline
```mermaid
flowchart TB
    subgraph Client["Client Interface"]
        Upload["Document Upload"]
        Query["User Query"]
        History["Chat History"]
    end

    subgraph DocumentProcessing["Document Processing Pipeline"]
        PDF["PDF Processing"]
        subgraph ImageProcessing["Image Processing"]
            ImgExtract["Image Extraction"]
            Caption["Caption Detection"]
            ImgAssoc["Image-Text Association"]
        end
        subgraph TextProcessing["Text Processing"]
            Hierarch["Hierarchical Chunking"]
            Layout["Layout Analysis"]
            BBDetect["Bounding Box Detection"]
        end
    end

    subgraph LLMProviders["LLM Providers"]
        ChatGPT["ChatGPT"]
        Ollama["Ollama"]
    end

    subgraph Indexing["Indexing & Storage"]
        subgraph VectorDB["Vector Storage (ChromaDB)"]
            Embed["Embedding Generation"]
            Store["Vector Storage"]
        end
        subgraph SparseIndex["Sparse Index"]
            BM25["BM25 Index"]
        end
        subgraph FileStore["File Storage"]
            PDFStore["PDF Storage"]
            ImgStore["Image Storage"]
            MetaStore["Metadata Storage"]
        end
    end

    subgraph QueryProcessing["Query Analysis"]
        Rephrase["Query Rephrase"]
        Decomp["Query Decomposition"]
        StepBack["Step-Back Prompting"]
        SubQueries["Sub-Query Generation"]
    end

    subgraph Retrieval["Hybrid Retrieval"]
        Dense["Dense Retrieval"]
        Sparse["Sparse Retrieval"]
        RRF["Reciprocal Rank Fusion"]
    end

    subgraph ResponseGen["Response Generation"]
        Context["Context Assembly"]
        ImgRef["Image Reference"]
        Citation["Citation Tracking"]
        Answer["Answer Generation"]
        Confidence["Confidence Score"]
    end

%% Document Processing Flow
    Upload --> PDFStore
    PDFStore --> PDF
    PDF --> ImgExtract
    PDF --> Hierarch

%% Image Processing Flow
    ImgExtract --> Caption
    ImgExtract --> ImgStore
    Caption --> ImgAssoc
    ImgAssoc --> MetaStore

%% Text Processing Flow
    Hierarch --> BBDetect
    BBDetect --> Layout
    Layout --> MetaStore

%% Indexing Flow
    Hierarch --> Embed
    Hierarch --> BM25
    Embed --> Store

%% LLM Provider Connections
    ChatGPT --> Embed
    Ollama --> Embed
    ChatGPT --> Answer
    Ollama --> Answer
    ChatGPT --> Decomp
    Ollama --> Decomp
    ChatGPT --> StepBack
    Ollama --> StepBack
    ChatGPT --> Rephrase
    Ollama --> Rephrase
    ChatGPT --> SubQueries
    Ollama --> SubQueries

%% Query Processing Flow
    Query --> Rephrase
    History --> Rephrase
    Rephrase --> Decomp
    Rephrase --> StepBack
    Decomp --> SubQueries

%% Retrieval Flow
    Store --> Dense
    BM25 --> Sparse
    SubQueries --> Dense
    SubQueries --> Sparse
    StepBack --> Dense
    StepBack --> Sparse
    Dense --> RRF
    Sparse --> RRF
    RRF --> Context

%% Response Generation Flow
    Context --> Citation
    MetaStore --> Citation
    ImgStore --> ImgRef
    MetaStore --> ImgRef

    Context --> Answer
    ImgRef --> Answer
    Citation --> Answer

%% Confidence generated alongside Answer
    Context --> Confidence
    Citation --> Confidence
    ImgRef --> Confidence
    Confidence --> Answer

%% Styling
    classDef primary fill:#ffffff,stroke:#000000,color:#000000
    classDef secondary fill:#cccccc,stroke:#000000,color:#000000
    classDef tertiary fill:#999999,stroke:#000000,color:#000000
    classDef quaternary fill:#666666,stroke:#000000,color:#ffffff
    classDef quinary fill:#333333,stroke:#000000,color:#ffffff

    class Upload,Query,History primary
    class PDF,Extract,Chunk,ImageProcessing,TextProcessing secondary
    class VectorDB,SparseIndex,FileStore,LLMProviders tertiary
    class Rephrase,Decomp,StepBack,SubQueries,Dense,Sparse,RRF quaternary
    class Context,ImgRef,Citation,Answer,Confidence quinary
```


## Detailed Features

### Document Understanding
- **Layout Analysis**: Preserves spatial relationships between text and visual elements
- **Image Processing**: Automatically extracts and associates images with relevant text
- **Table Detection**: Identifies and preserves tabular data structures
- **Metadata Tracking**: Maintains document structure and relationships


### Query Processing
- **Chat History Analysis**: Rephrases queries based on conversation context
- **Query Decomposition**: Breaks complex queries into manageable sub-queries
- **Context Expansion**: Uses step-back prompting for broader understanding
- **Multi-stage Retrieval**: Combines results from multiple retrieval strategies
- **Citation Tracking**: Maintains source attribution for all retrieved information


### Response Generation
- **Context Assembly**: Intelligently combines retrieved information
- **Image Integration**: Seamlessly incorporates relevant images and diagrams
- **Citation Management**: Provides detailed source tracking
- **Confidence Scoring**: Evaluates response reliability

![Response Example](images/response_example.png)

## Important Usage Notes

### Document Processing

**⚠️ Important**: Document embeddings are provider-specific and are generated at upload time based on your current configuration.

- Documents are embedded using all currently configured providers
- Embedding model changes require document re-upload
- Provider configuration should precede document upload

### Storage Requirements
- Recommended: 15GB+ free space for document storage
- Additional space needed for embedding storage
- Consider storage requirements when processing large documents

## Configuration Guide

### OpenAI Setup
1. Obtain API key from platform.openai.com
2. Configure model and embedding model
3. Set temperature and other parameters
4. Save configuration

### Ollama Setup
1. Ensure Ollama is running (default: http://ollama:11434)
2. Select model (e.g., llama2, mistral)
3. Configure embedding model (e.g., nomic-embed-text)
4. Save configuration

## Performance Considerations

### Response Time Optimization
- Hybrid retrieval balances speed and accuracy
- Chunk size affects retrieval precision
- Provider selection impacts latency and cost

### Memory Management
- Monitor embedding storage growth
- Consider regular maintenance for optimal performance
- Balance chunk size with retrieval effectiveness

## Use Cases

### Document Analysis
- Research paper analysis
- Technical documentation understanding
- Multi-modal content processing

### Information Retrieval
- Complex query resolution
- Multi-document synthesis
- Visual information integration

### Comparative Analysis
- Provider performance evaluation
- Embedding model comparison
- Retrieval strategy optimization

## Technical Details

### Backend Architecture
- FastAPI for robust API handling
- Async processing for improved performance
- Modular design for provider integration

### Storage Layer
- ChromaDB for vector storage
- PostgreSQL for metadata and system state
- File system for document storage

### Frontend
- React with Next.js
- Real-time UI updates
- Responsive design

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CLA.md) before submitting changes.

## License
This project is licensed under a Custom Non-Commercial License. See [LICENSE](LICENSE.md) for full details.

## Acknowledgments
- Anthropic, OpenAI, and the Ollama team for their excellent models
- The open-source community for various supporting libraries
- All contributors who have helped improve this project

## Support My Work

If you find EchoQuest useful and want to help me keep developing innovative, open-source tools, consider supporting me by buying me a token. Your support helps cover development costs and allows me to create more projects like this!

[Buy me a token!](https://buymeacoffee.com/TobiFank)

Or, scan the QR code below to contribute:

![Buy me a token QR Code](images/buymeatokenqr.png)

Thank you for your support! It truly makes a difference.

