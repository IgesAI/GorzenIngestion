# Gorzen Ingestion - Universal Document Vector Pipeline

Transform any document collection into a searchable vector database in minutes! A powerful, user-friendly web application that combines Docling's advanced PDF processing with Pinecone's vector storage capabilities.

## 🚀 Features

- **🌐 Web Interface**: Beautiful, intuitive drag-and-drop interface
- **📄 Multi-Format Support**: PDF, DOCX, TXT, and Markdown files
- **🔧 Zero Configuration**: Just upload files and enter your API keys
- **⚡ Real-time Processing**: Live progress tracking and status updates
- **🎯 Smart Chunking**: Configurable chunking strategies for optimal search
- **🤖 Dual Embedding Options**: Local models or OpenAI embeddings
- **📊 Rich Metadata**: Automatic document classification and extraction
- **☁️ Cloud Ready**: Deploy instantly on Vercel or any Node.js platform

## 🚀 Quick Start

### Option 1: Deploy to Vercel (Recommended)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/IgesAI/GorzenIngestion)

1. Click the deploy button above
2. Connect your GitHub account
3. Set your environment variables (see Configuration section)
4. Deploy and start processing documents!

### Option 2: Local Development

1. Clone and install:
```bash
git clone https://github.com/IgesAI/GorzenIngestion.git
cd GorzenIngestion
npm install
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp env.example .env.local
# Edit .env.local with your API keys
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🛠️ How It Works

1. **📤 Upload**: Drag and drop your documents (PDF, DOCX, TXT, MD)
2. **⚙️ Configure**: Enter your Pinecone API key and choose settings
3. **🔄 Process**: Watch real-time progress as documents are converted and indexed
4. **✅ Complete**: Your vector database is ready for semantic search!

## 🔧 Configuration

### Environment Variables

For deployment, set these environment variables:

```bash
# Required: Pinecone API key
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: OpenAI API key (for better embeddings)
OPENAI_API_KEY=your_openai_api_key_here
```

### API Keys Setup

1. **Pinecone**: Get your API key from [Pinecone Console](https://app.pinecone.io/)
2. **OpenAI** (optional): Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 📖 Usage Guide

### Web Interface (Recommended)

1. Open the application in your browser
2. Drag & drop your documents onto the upload area
3. Configure your Pinecone settings
4. Click "Start Processing" and watch the magic happen!

### Command Line Interface

For batch processing or automation:

```bash
# Basic usage
python ingest.py --source docs/ --index my-index

# With OpenAI embeddings
python ingest.py --source docs/ --index my-index --use-openai

# With enrichments
python ingest.py --source docs/ --index my-index --enrich-picture-description
```

## 🎯 Features Deep Dive

### Chunking Strategies
- **Small (512 tokens)**: Best for precise search and Q&A
- **Medium (1024 tokens)**: Balanced performance (recommended)
- **Large (1536 tokens)**: Better context for complex documents

### Embedding Options
- **Local Models**: Free, runs offline using BAAI/bge-small-en-v1.5
- **OpenAI**: Higher quality, requires API key, uses text-embedding-3-small

### Document Enrichments
- **Code Understanding**: Better processing of code blocks
- **Formula Recognition**: Enhanced mathematical content handling
- **Image Classification**: Automatic image categorization
- **Image Descriptions**: AI-generated descriptions of diagrams and figures

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

## 🚀 Deployment

### Vercel (Recommended)

1. Fork this repository
2. Connect your GitHub to Vercel
3. Set environment variables in Vercel dashboard:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY` (optional)
4. Deploy!

### Other Platforms

- **Railway**: Connect GitHub repo, set env vars, deploy
- **Render**: Same process as Railway
- **DigitalOcean App Platform**: Upload repo, configure build settings

## 🛠️ Development

### Project Structure
```
GorzenIngestion/
├── app/                    # Next.js app directory
│   ├── components/         # React components
│   ├── api/               # API routes
│   └── page.tsx           # Main page
├── lib/                   # Shared utilities
├── ingest.py             # Core Python processing script
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies
└── next.config.js        # Next.js configuration
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and test
4. Submit a pull request

## 📄 License

This project is proprietary software developed by IgesAI.

## 🤝 Support

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed information for faster resolution
