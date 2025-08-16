#!/usr/bin/env python3
"""
Test script to validate Pinecone connection and basic functionality.
Run this before using the main ingestion script to ensure everything works.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import pinecone
        print(f"✓ Pinecone SDK {pinecone.__version__}")
    except ImportError as e:
        print(f"✗ Pinecone import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✓ SentenceTransformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"✗ SentenceTransformers import failed: {e}")
        return False
        
    try:
        import docling
        print(f"✓ Docling available")
    except ImportError as e:
        print(f"✗ Docling import failed: {e}")
        return False
    
    try:
        import openai
        print(f"✓ OpenAI {openai.__version__}")
    except ImportError as e:
        print(f"⚠ OpenAI not available (optional): {e}")
    
    return True

def test_pinecone_connection():
    """Test Pinecone API connection."""
    print("\nTesting Pinecone connection...")
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("✗ PINECONE_API_KEY not set in environment")
        return False
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Test connection by listing indexes
        indexes = pc.list_indexes()
        print(f"✓ Pinecone connection successful")
        print(f"  Available indexes: {len(indexes.names()) if hasattr(indexes, 'names') else 'unknown'}")
        
        if hasattr(indexes, 'names') and indexes.names():
            for idx_name in indexes.names()[:3]:  # Show first 3
                print(f"    - {idx_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pinecone connection failed: {e}")
        return False

def test_embedding_model():
    """Test local embedding model loading."""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully")
        print(f"  Dimension: {dimension}")
        
        # Test encoding
        test_text = ["This is a test sentence for embedding."]
        embeddings = model.encode(test_text, normalize_embeddings=True)
        actual_dim = len(embeddings[0])
        
        print(f"  Test encoding: {actual_dim}-dimensional vector")
        
        if actual_dim != dimension:
            print(f"⚠ Dimension mismatch: reported {dimension}, actual {actual_dim}")
        else:
            print("✓ Embedding test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding model test failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection (optional)."""
    print("\nTesting OpenAI connection...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set (optional)")
        return True
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a small request
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"
        )
        
        embedding = response.data[0].embedding
        print(f"✓ OpenAI connection successful")
        print(f"  Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Gorzen Ingestion - System Test")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Run tests
    tests = [
        test_imports,
        test_pinecone_connection,
        test_embedding_model,
        test_openai_connection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for document ingestion.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
