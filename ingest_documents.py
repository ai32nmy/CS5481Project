"""
Document Ingestion Script
Processes documents and creates/updates the vector database.
"""

import sys
# Suppress broken tensorflow installation
sys.modules['tensorflow'] = None

import argparse
import os
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager


def main():
    """Main function for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the knowledge base"
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help="Directory containing documents (uses config default if not specified)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector database (delete existing data)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check for API keys based on configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    provider = config['embeddings']['provider']
    if provider == 'gemini' and not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment variables.")
        print("Please set this in a .env file to use Google Gemini embeddings.")
    elif provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set this in a .env file to use OpenAI embeddings.")
    elif provider == 'huggingface':
        print(f"Using local HuggingFace embeddings: {config['embeddings']['huggingface_model']}")
        print("No API key required - models will be downloaded locally on first use.")

    print("="*60)
    print("Document Ingestion Pipeline")
    print("="*60)

    try:
        # Initialize processor and vector store
        processor = DocumentProcessor(args.config)
        vector_store_manager = VectorStoreManager(args.config)

        # Process documents
        print("\nStep 1: Processing documents...")
        chunks = processor.process_documents(args.documents_dir)

        if not chunks:
            print("No documents to ingest. Please add documents to the documents directory.")
            return

        print(f"\nProcessed {len(chunks)} document chunks")

        # Handle reset option
        if args.reset:
            import shutil
            persist_dir = vector_store_manager.persist_directory
            if os.path.exists(persist_dir):
                print(f"\nRemoving existing vector database at {persist_dir}...")
                shutil.rmtree(persist_dir)

        # Create or update vector store
        print("\nStep 2: Creating vector embeddings and storing in database...")
        print("(This may take a few minutes depending on the number of documents)")

        if args.reset or not os.path.exists(vector_store_manager.persist_directory):
            vector_store_manager.create_vector_store(chunks)
        else:
            print("Existing vector store found. Adding new documents...")
            vector_store_manager.add_documents(chunks)

        print("\n" + "="*60)
        print("Ingestion Complete!")
        print("="*60)
        print(f"Total chunks in database: {len(chunks)}")
        print(f"Vector database location: {vector_store_manager.persist_directory}")
        print("\nYou can now run the RAG system with: python main.py")

    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
