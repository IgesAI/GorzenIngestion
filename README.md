# Gorzen Document Ingestion

A powerful document ingestion pipeline for technical manuals using Docling for PDF conversion, advanced chunking, and Pinecone for vector storage.

## Features

- **Advanced PDF Processing**: Uses Docling for high-quality PDF conversion with optional enrichment features
- **Smart Chunking**: Configurable hybrid chunking with contextual text generation
- **Flexible Embeddings**: Support for both local SentenceTransformers and OpenAI embedding models
- **Rich Metadata**: Automatic document classification and structured metadata extraction
- **Batch Processing**: Efficient batch processing with progress tracking
- **Vector Storage**: Seamless integration with Pinecone for scalable vector search

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IgesAI/GorzenIngestion.git
cd GorzenIngestion
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

```bash
# Required: Pinecone API key
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: OpenAI API key (if using OpenAI embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Pinecone settings
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Optional: Docling artifacts path for offline model usage
DOCLING_ARTIFACTS_PATH=/path/to/docling/models
```

## Usage

### Basic Usage

Process a single PDF:
```bash
python ingest.py --source docs/manual.pdf --index my-index
```

Process all PDFs in a directory:
```bash
python ingest.py --source docs/ --index my-index
```

### Advanced Options

#### Chunk Size Configuration
```bash
# Small chunks (512 tokens) - good for precise search
python ingest.py --source docs/ --index my-index --chunk-size small

# Medium chunks (1024 tokens) - balanced performance
python ingest.py --source docs/ --index my-index --chunk-size medium

# Large chunks (1536 tokens) - better for complex context
python ingest.py --source docs/ --index my-index --chunk-size large
```

#### Embedding Models
```bash
# Use local SentenceTransformers model (default)
python ingest.py --source docs/ --index my-index --model BAAI/bge-small-en-v1.5

# Use OpenAI embeddings
python ingest.py --source docs/ --index my-index --use-openai
```

#### Document Enrichment
```bash
# Enable all enrichment features
python ingest.py --source docs/ --index my-index \
  --enrich-code \
  --enrich-formula \
  --enrich-picture-classes \
  --enrich-picture-description
```

#### Metadata and Debugging
```bash
# Show sample metadata for each document
python ingest.py --source docs/ --index my-index --show-metadata

# Custom batch size for processing
python ingest.py --source docs/ --index my-index --batch-size 32
```

### Complete Example

```bash
python ingest.py \
  --source docs/ \
  --index technical-manuals \
  --chunk-size medium \
  --use-openai \
  --enrich-picture-description \
  --show-metadata \
  --batch-size 64 \
  --prefix manual
```

## Document Types Supported

The system automatically classifies documents based on filename patterns:

- **User Manuals**: Files containing "manual"
- **Service Procedures**: Files with "install", "replacement", "swap"
- **Maintenance Guides**: Files with "care", "washing"
- **Firmware Guides**: Files with "flash", "firmware"
- **Charger Manuals**: Files with "charger"
- **Diagnostic Tools**: Files with "scope"
- **Reference Guides**: Files with "tips", "definitions"
- **Battery Guides**: Files with "battery"

## Metadata Structure

Each chunk includes rich metadata for enhanced search and filtering:

```json
{
  "document_name": "CX5E2025MANUAL",
  "document_type": "user_manual",
  "document_category": "technical",
  "source_file": "CX5E2025MANUAL rev 11.pdf",
  "model": "CX5E",
  "version": "Rev 11",
  "page": 15,
  "chunk_id": 3,
  "total_chunks": 45,
  "content_type": "procedure",
  "is_procedure": true,
  "is_safety": false,
  "has_images": true,
  "primary_heading": "Motor Installation",
  "text": "actual chunk content..."
}
```

## Architecture

```
PDF Documents → Docling Conversion → Smart Chunking → Embedding Generation → Pinecone Storage
     ↓                ↓                    ↓                    ↓                 ↓
- Technical PDFs   - Layout detection   - Hybrid chunker   - Local or OpenAI   - Vector search
- User manuals     - Table extraction   - Context aware    - Normalized       - Metadata filters  
- Service docs     - Image processing   - Configurable     - Batch processing - Scalable storage
```

## Development

### Project Structure
```
GorzenIngestion/
├── ingest.py          # Main ingestion script
├── requirements.txt   # Python dependencies
├── README.md         # This file
├── .env.example      # Environment template
├── .gitignore        # Git ignore rules
└── docs/             # Sample documents (not in repo)
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to your fork: `git push origin feature-name`
6. Create a pull request

## License

This project is proprietary software developed by IgesAI.

## Support

For issues and questions, please create an issue in the GitHub repository.
