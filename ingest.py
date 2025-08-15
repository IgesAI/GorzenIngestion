import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

import click
from dotenv import load_dotenv
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


def yield_pdf_paths(source: Path) -> Iterable[Path]:
    """Yield PDF file paths from a file or directory."""
    if source.is_file() and source.suffix.lower() == ".pdf":
        yield source
    elif source.is_dir():
        for p in sorted(source.rglob("*.pdf")):
            yield p
    else:
        raise FileNotFoundError(f"No PDF file(s) found at: {source}")


def build_converter(
    enrich_code: bool,
    enrich_formula: bool,
    enrich_picture_classes: bool,
    enrich_picture_description: bool,
    artifacts_path: str | None,
) -> DocumentConverter:
    """Build DocumentConverter with optional enrichment features."""
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path if artifacts_path else None
    )

    # Always generate picture images for better chunking
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0  # Higher quality images
    
    if enrich_code:
        pipeline_options.do_code_enrichment = True
    if enrich_formula:
        pipeline_options.do_formula_enrichment = True
    if enrich_picture_classes:
        pipeline_options.do_picture_classification = True
    if enrich_picture_description:
        from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions
        pipeline_options.do_picture_description = True
        # Use a custom prompt for better technical descriptions
        pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",  # Better than default
            prompt="Describe this technical diagram or image in detail. Focus on any visible components, wires, connectors, parts, labels, measurements, or technical details. Be specific about what you can see.",
        )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def docling_convert(converter: DocumentConverter, pdf_path: Path) -> Any:
    """Convert PDF using Docling."""
    return converter.convert(str(pdf_path)).document


def chunk_document(dl_doc: Any, tokenizer_model: str, chunk_size: str = "medium") -> Iterable[Dict[str, Any]]:
    """Chunk Docling document using HybridChunker with configurable sizes."""
    
    # Chunk size configurations for different use cases
    size_configs = {
        "small": {"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 50},
        "medium": {"max_tokens": 1024, "min_tokens": 200, "overlap_tokens": 100},  # Best for technical chatbots
        "large": {"max_tokens": 1536, "min_tokens": 300, "overlap_tokens": 150},   # For complex technical docs
    }
    
    config = size_configs[chunk_size]
    click.echo(f"Using {chunk_size} chunks: {config['max_tokens']} max tokens, {config['min_tokens']} min tokens")
    
    chunker = HybridChunker(
        tokenizer=tokenizer_model,
        max_tokens=config["max_tokens"],
        min_tokens=config["min_tokens"],
        overlap_tokens=config["overlap_tokens"],
        merge_across_headers=True,  # Better content flow
        merge_across_pages=False,  # Keep page boundaries clear
    )
    for chunk in chunker.chunk(dl_doc):
        # Use contextualized text for better quality
        text: str = chunker.contextualize(chunk=chunk)
        
        # Extract metadata from the chunk
        page_no = None
        headings = []
        doc_items_refs = []
        
        try:
            # Access doc_items from chunk.meta
            if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
                for doc_item in chunk.meta.doc_items:
                    # Collect references
                    if hasattr(doc_item, 'self_ref'):
                        doc_items_refs.append(doc_item.self_ref)
                    
                    # Extract page number from provenance
                    if hasattr(doc_item, 'prov') and doc_item.prov:
                        for prov in doc_item.prov:
                            if hasattr(prov, 'page_no') and prov.page_no is not None:
                                page_no = prov.page_no
                                break
            
            # Extract headings if available
            if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings'):
                headings = chunk.meta.headings
                
        except Exception as e:
            # Fallback - just use the raw chunk text
            if hasattr(chunk, 'text'):
                text = chunk.text
            pass

        yield {
            "text": text,
            "meta": {
                "page": page_no,
                "headings": headings,
                "doc_items_refs": doc_items_refs,
            },
        }


def ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str, cloud: str, region: str):
    """Ensure Pinecone index exists, create if missing."""
    try:
        idx = pc.Index(index_name)
        _ = idx.describe_index_stats()
        return idx
    except Exception:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        return pc.Index(index_name)


def embed_texts(model, texts: List[str], use_openai: bool = False) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    if use_openai:
        # OpenAI embeddings
        response = model.embeddings.create(
            input=texts,
            model=OPENAI_EMBED_MODEL
        )
        return [embedding.embedding for embedding in response.data]
    else:
        # SentenceTransformers embeddings
        emb = model.encode(texts, normalize_embeddings=True)
        return emb.tolist()


def upsert_vectors(index, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
    """Upsert vectors to Pinecone index."""
    items = []
    for id_, vec, md in zip(ids, vectors, metadatas):
        items.append({"id": id_, "values": vec, "metadata": md})
    index.upsert(vectors=items)


@click.command()
@click.option("--source", "source_path", required=True, type=click.Path(exists=True, path_type=Path), 
              help="Path to a PDF file or directory containing PDFs")
@click.option("--index", "index_name", required=True, 
              help="Pinecone index name")
@click.option("--model", "embed_model_name", default=DEFAULT_EMBED_MODEL, show_default=True,
              help="SentenceTransformers embedding model")
@click.option("--use-openai/--no-use-openai", default=False, show_default=True,
              help="Use OpenAI text-embedding-3-small (1536 dims) instead of local model")
@click.option("--chunk-size", type=click.Choice(["small", "medium", "large"]), default="medium", show_default=True,
              help="Chunk size: small(512), medium(1024), large(1536) tokens")
@click.option("--show-metadata/--no-show-metadata", default=False, show_default=True,
              help="Show sample metadata for first chunk of each document")
@click.option("--batch-size", default=64, show_default=True,
              help="Embedding/upsert batch size")
@click.option("--cloud", default=lambda: os.getenv("PINECONE_CLOUD", "aws"), 
              show_default="env PINECONE_CLOUD or 'aws'")
@click.option("--region", default=lambda: os.getenv("PINECONE_REGION", "us-east-1"), 
              show_default="env PINECONE_REGION or 'us-east-1'")
@click.option("--metric", default="cosine", show_default=True)
@click.option("--text-metadata-key", default="text", show_default=True,
              help="Metadata key storing original chunk text")
@click.option("--prefix", default=None, 
              help="Optional ID prefix for vectors")
@click.option("--enrich-code/--no-enrich-code", default=False, show_default=True,
              help="Enable code understanding enrichment")
@click.option("--enrich-formula/--no-enrich-formula", default=False, show_default=True,
              help="Enable formula understanding enrichment")
@click.option("--enrich-picture-classes/--no-enrich-picture-classes", default=False, show_default=True,
              help="Enable picture classification enrichment")
@click.option("--enrich-picture-description/--no-enrich-picture-description", default=False, show_default=True,
              help="Enable picture description enrichment")
@click.option("--artifacts-path", default=lambda: os.getenv("DOCLING_ARTIFACTS_PATH"), 
              show_default="env DOCLING_ARTIFACTS_PATH", help="Prefetched Docling models path")
def main(
    source_path: Path,
    index_name: str,
    embed_model_name: str,
    use_openai: bool,
    chunk_size: str,
    show_metadata: bool,
    batch_size: int,
    cloud: str,
    region: str,
    metric: str,
    text_metadata_key: str,
    prefix: str | None,
    enrich_code: bool,
    enrich_formula: bool,
    enrich_picture_classes: bool,
    enrich_picture_description: bool,
    artifacts_path: str | None,
):
    """Convert PDFs with Docling, chunk, embed, and upsert to Pinecone."""
    load_dotenv()

    # Get Pinecone API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise click.UsageError("PINECONE_API_KEY not set. Check your .env file.")

    pc = Pinecone(api_key=api_key)

    # Load embedding model
    if use_openai:
        if not OPENAI_AVAILABLE:
            raise click.UsageError("OpenAI not installed. Run: pip install openai")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise click.UsageError("OPENAI_API_KEY not set. Add it to your .env file.")
        
        click.echo(f"Using OpenAI embedding model: {OPENAI_EMBED_MODEL}")
        emb_model = OpenAI(api_key=openai_api_key)
        dimension = 1536  # text-embedding-3-small dimension
    else:
        click.echo(f"Loading local embedding model: {embed_model_name}")
        emb_model = SentenceTransformer(embed_model_name)
        dimension = emb_model.get_sentence_embedding_dimension()
    
    click.echo(f"Embedding dimension: {dimension}")

    # Ensure Pinecone index exists
    index = ensure_index(pc, index_name=index_name, dimension=dimension, metric=metric, cloud=cloud, region=region)
    click.echo(f"Using Pinecone index: {index_name} ({cloud}/{region})")

    # Build Docling converter
    converter = build_converter(
        enrich_code=enrich_code,
        enrich_formula=enrich_formula,
        enrich_picture_classes=enrich_picture_classes,
        enrich_picture_description=enrich_picture_description,
        artifacts_path=artifacts_path,
    )

    # Find PDF files to process
    source_path = source_path.resolve()
    pdf_paths = list(yield_pdf_paths(source_path))
    if not pdf_paths:
        raise click.ClickException(f"No PDF files found at {source_path}")

    # Process each PDF
    for pdf in pdf_paths:
        click.echo(f"Processing: {pdf}")
        
        # Convert with Docling
        dl_doc = docling_convert(converter, pdf)
        
        # Chunk the document
        chunks = list(chunk_document(dl_doc, tokenizer_model=embed_model_name, chunk_size=chunk_size))
        if not chunks:
            click.echo("No chunks produced; skipping.")
            continue

        # Show chunk statistics
        chunk_lengths = [len(c["text"]) for c in chunks]
        click.echo(f"Generated {len(chunks)} chunks:")
        click.echo(f"  Avg length: {sum(chunk_lengths)/len(chunk_lengths):.0f} chars")
        click.echo(f"  Min length: {min(chunk_lengths)} chars")
        click.echo(f"  Max length: {max(chunk_lengths)} chars")
        
        # Prepare texts and metadata with rich document information
        texts: List[str] = [c["text"] for c in chunks]
        metas: List[Dict[str, Any]] = []
        
        # Extract document-level metadata
        doc_name = pdf.stem
        doc_type = "unknown"
        doc_category = "technical"
        
        # Classify document type based on filename
        filename_lower = doc_name.lower()
        if "manual" in filename_lower:
            doc_type = "user_manual"
        elif "install" in filename_lower or "replacement" in filename_lower or "swap" in filename_lower:
            doc_type = "service_procedure"
        elif "care" in filename_lower or "washing" in filename_lower:
            doc_type = "maintenance_guide"
        elif "flash" in filename_lower or "firmware" in filename_lower:
            doc_type = "firmware_guide"
        elif "charger" in filename_lower:
            doc_type = "charger_manual"
        elif "scope" in filename_lower:
            doc_type = "diagnostic_tool"
        elif "tips" in filename_lower or "definitions" in filename_lower:
            doc_type = "reference_guide"
        elif "battery" in filename_lower:
            doc_type = "battery_guide"
        
        # Extract model/version info
        model_info = ""
        version_info = ""
        if "CX" in doc_name:
            model_match = [part for part in doc_name.split() if part.startswith("CX")]
            if model_match:
                model_info = model_match[0]
        elif "FWE" in doc_name:
            model_info = "FWE"
        elif "JR" in doc_name:
            model_info = "JR"
        elif "SR" in doc_name:
            model_info = "SR"
        elif "652025" in doc_name:
            model_info = "652025"
        elif "652026" in doc_name:
            model_info = "652026"
            
        # Extract version/revision info
        if "rev" in filename_lower or "REV" in doc_name:
            version_parts = doc_name.split()
            for i, part in enumerate(version_parts):
                if part.lower() in ["rev", "REV"] and i + 1 < len(version_parts):
                    version_info = f"Rev {version_parts[i + 1]}"
                    break
        
        for i, c in enumerate(chunks):
            # Start fresh with only Pinecone-compatible metadata
            clean_meta = {}
            
            # Core document metadata (strings and numbers only)
            clean_meta["document_name"] = str(doc_name)
            clean_meta["document_type"] = str(doc_type)
            clean_meta["document_category"] = str(doc_category)
            clean_meta["source_file"] = str(pdf.name)
            clean_meta["chunk_id"] = int(i + 1)
            clean_meta["total_chunks"] = int(len(chunks))
            clean_meta["chunk_length"] = int(len(c["text"]))
            
            # Model and version (only if not empty)
            if model_info and model_info.strip():
                clean_meta["model"] = str(model_info)
            if version_info and version_info.strip():
                clean_meta["version"] = str(version_info)
            
            # Page info from original metadata (ensure it's a number)
            try:
                page_no = c.get("meta", {}).get("page")
                if page_no is not None:
                    clean_meta["page"] = int(page_no)
            except (ValueError, TypeError):
                pass
            
            # Headings - convert to single primary heading string
            try:
                headings = c.get("meta", {}).get("headings", [])
                if headings and isinstance(headings, list):
                    # Get first non-empty heading
                    primary_heading = None
                    for h in headings:
                        if h and str(h).strip():
                            primary_heading = str(h).strip()
                            break
                    if primary_heading:
                        clean_meta["primary_heading"] = primary_heading
            except Exception:
                pass
            
            # Content type analysis
            chunk_text = c["text"].lower()
            if any(word in chunk_text for word in ["step", "procedure", "install", "remove", "replace"]):
                clean_meta["content_type"] = "procedure"
            elif any(word in chunk_text for word in ["warning", "caution", "danger", "important"]):
                clean_meta["content_type"] = "safety"
            elif any(word in chunk_text for word in ["specification", "rating", "voltage", "current", "torque"]):
                clean_meta["content_type"] = "specifications"
            elif any(word in chunk_text for word in ["troubleshoot", "problem", "error", "fault"]):
                clean_meta["content_type"] = "troubleshooting"
            elif "image" in chunk_text or "picture" in chunk_text or "diagram" in chunk_text:
                clean_meta["content_type"] = "visual"
            else:
                clean_meta["content_type"] = "general"
            
            # Boolean flags for easy filtering
            clean_meta["has_images"] = bool("image" in chunk_text or "picture" in chunk_text or "diagram" in chunk_text)
            clean_meta["is_procedure"] = bool(any(word in chunk_text for word in ["step", "procedure", "install", "remove", "replace"]))
            clean_meta["is_safety"] = bool(any(word in chunk_text for word in ["warning", "caution", "danger", "important"]))
            
            # Store text content (truncate if needed to stay under 40KB total metadata limit)
            text_content = c["text"]
            if len(text_content) > 30000:  # Leave room for other metadata
                text_content = text_content[:30000] + "..."
            clean_meta[text_metadata_key] = str(text_content)
            
            # Final validation: ensure all metadata is Pinecone-compatible
            validated_meta = {}
            for key, value in clean_meta.items():
                # Only include supported types
                if isinstance(value, str) and value.strip():
                    validated_meta[key] = value.strip()
                elif isinstance(value, (int, float)):
                    validated_meta[key] = value
                elif isinstance(value, bool):
                    validated_meta[key] = value
                # Skip anything else (None, empty strings, complex objects, etc.)
            
            metas.append(validated_meta)

        # Show sample metadata if requested
        if show_metadata and metas:
            click.echo(f"\n📋 Sample metadata for {doc_name}:")
            sample_meta = {k: v for k, v in metas[0].items() if k != text_metadata_key}
            for key, value in sample_meta.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                click.echo(f"  {key}: {value}")

        # Embed and upsert in batches
        doc_prefix = prefix or pdf.stem
        pbar = tqdm(total=len(texts), ncols=100, desc="Upserting")
        
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_metas = metas[start:end]
            
            # Generate embeddings
            vecs = embed_texts(emb_model, batch_texts, use_openai=use_openai)
            
            # Create unique IDs
            ids = [f"{doc_prefix}-{i}" for i in range(start, end)]
            
            # Upsert to Pinecone
            upsert_vectors(index, ids, vecs, batch_metas)
            pbar.update(len(batch_texts))
            
        pbar.close()
        click.echo(f"Processed {len(chunks)} chunks from {pdf}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
