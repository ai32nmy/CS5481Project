"""
Main Application
Interactive CLI for the Agentic RAG system.
"""

import sys
# Suppress broken tensorflow installation
sys.modules['tensorflow'] = None

import argparse
import os
import sys
from dotenv import load_dotenv

from src.agent import AgenticRAG


def print_header():
    """Print the application header."""
    print("\n" + "="*60)
    print(" " * 15 + "AGENTIC RAG SYSTEM")
    print(" " * 10 + "Knowledge-Based Q&A with AI Agent")
    print("="*60)


def print_help():
    """Print help information."""
    help_text = """
Available Commands:
  - Type your question to get an answer from the knowledge base
  - /help    : Show this help message
  - /clear   : Clear conversation history
  - /stats   : Show knowledge base statistics
  - /quit    : Exit the application (or Ctrl+C)
"""
    print(help_text)


def main():
    """Main function for the RAG application."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG System - Interactive Q&A"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Ask a single question and exit (non-interactive mode)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config and check for appropriate API key
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check for API key based on LLM provider (skip for local models)
    llm_provider = config['llm']['provider']
    if llm_provider == 'gemini' and not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your Google Gemini API key.")
        print("See .env.example for reference.")
        return 1
    elif llm_provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("See .env.example for reference.")
        return 1
    elif llm_provider == 'ollama':
        print(f"Using Ollama with model: {config['llm']['ollama_model']}")
        print(f"Make sure Ollama is running at: {config['llm'].get('ollama_base_url', 'http://localhost:11434')}")

    # Check if vector database exists
    persist_dir = config['vector_db']['persist_directory']

    if not os.path.exists(persist_dir):
        print("Error: Vector database not found!")
        print("Please run the ingestion script first:")
        print("  python ingest_documents.py")
        return 1

    try:
        # Initialize the agent
        print("Initializing Agentic RAG System...")
        agent = AgenticRAG(args.config)

        # Single question mode
        if args.question:
            print(f"\nQuestion: {args.question}")
            response = agent.query(args.question)
            print(f"\nAnswer: {response}\n")
            return 0

        # Interactive mode
        print_header()
        print_help()

        print("\nAgent is ready! Ask me anything about the knowledge base.")
        print("Type /quit to exit.\n")

        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()

                    if command == '/quit' or command == '/exit':
                        print("\nGoodbye!")
                        break

                    elif command == '/help':
                        print_help()
                        continue

                    elif command == '/clear':
                        agent.reset_memory()
                        print("Conversation history cleared.")
                        continue

                    elif command == '/stats':
                        response = agent.query("Show me the knowledge base statistics")
                        print(f"\n\033[1;32mAgent:\033[0m {response}")
                        continue

                    else:
                        print(f"Unknown command: {command}")
                        print("Type /help for available commands.")
                        continue

                # Process question
                print("\n\033[1;32mAgent:\033[0m Thinking...", end="\r")
                response = agent.query(user_input)
                print(f"\033[1;32mAgent:\033[0m {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or type /quit to exit.")

    except Exception as e:
        print(f"\nFatal Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
