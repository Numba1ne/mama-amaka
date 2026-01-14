import os
import glob
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import VectorDB - use relative import since both files are in src/
from vectordb import VectorDB
from memory import ConversationMemory

# Import LLM providers
try:
    from langchain_openai import ChatOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_groq import ChatGroq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MamaAmakaAgent:
    def __init__(self):
        print("Initializing Mama Amaka Agent...")
        self.vector_db = VectorDB(collection_name="mama_amaka_recipes")

        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=10)

        # Initialize LLM with multi-provider support
        self.llm = self._initialize_llm()
        if not self.llm:
            print("Error: No valid API key found. Please set one of:")
            print("  - OPENAI_API_KEY")
            print("  - GROQ_API_KEY")
            print("  - GOOGLE_API_KEY")
            print("in your .env file or environment variables.")
            sys.exit(1)

        # Create RAG prompt template with conversation history (Step 6 from template)
        prompt_template_str = """You are Mama Amaka, a warm, experienced Nigerian mother and cook. 
You answer questions about Nigerian cooking using the provided recipe context.

{conversation_history}

Use the following context from your cookbook to answer the question. 
If the context doesn't contain enough information to answer the question, 
say you don't recall that specific recipe right now, but be friendly and encourage them to cook!

Context:
{context}

Question: {question}

Answer as Mama Amaka:"""

        self.prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

        # Create the chain (prompt -> LLM -> output parser)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("Mama Amaka is ready!")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"  - Using OpenAI model: {model_name}")
            return ChatOpenAI(
                model=model_name,
                temperature=0.7,  # Slightly higher for more personality
            )

        # Check for Groq API key
        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"  - Using Groq model: {model_name}")
            return ChatGroq(model=model_name, temperature=0.7)

        # Check for Google API key
        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"  - Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.7)

        return None

    def load_recipes(self, data_dir: str = "data") -> List[Dict]:
        """Loads all text files from the data directory."""
        documents = []
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory '{data_dir}' not found.")
            return []

        # Find all .txt files
        file_paths = glob.glob(os.path.join(data_dir, "*.txt"))

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        filename = os.path.basename(file_path)
                        documents.append(
                            {"content": content, "metadata": {"source": filename}}
                        )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return documents

    def ingest_data(self):
        """Loads and indexes data."""
        print("\n--- Ingesting Recipes ---")
        docs = self.load_recipes()
        if docs:
            print(f"Found {len(docs)} recipes. Adding to VectorDB...")
            self.vector_db.add_documents(docs)
            print("Ingestion complete.")
        else:
            print("No recipes found to ingest.")

    def ask(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG assistant with enhanced retrieval evaluation and conversation memory (Step 7 from template).

        Args:
            query: User's question
            n_results: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold (0.0-1.0, lower is more similar)
            verbose: Whether to print detailed retrieval information

        Returns:
            Dictionary containing:
                - answer: Generated response string
                - sources: List of source documents used
                - similarity_scores: List of similarity scores for retrieved chunks
                - num_chunks: Number of chunks retrieved
        """
        if verbose:
            print(f"\nSearching for: '{query}'...")

        # 1. Retrieve relevant context chunks
        search_results = self.vector_db.search(query, n_results=n_results)
        documents = search_results.get("documents", [])
        metadatas = search_results.get("metadatas", [])
        distances = search_results.get("distances", [])

        # Convert distances to similarity scores (ChromaDB uses distance, lower is better)
        # Normalize to 0-1 scale where 1 is most similar
        similarity_scores = []
        if distances:
            max_dist = max(distances) if distances else 1.0
            similarity_scores = [
                1.0 - (d / max_dist) if max_dist > 0 else 1.0 for d in distances
            ]

        # Filter by minimum similarity threshold
        filtered_docs = []
        filtered_metas = []
        filtered_scores = []

        for i, (doc, score) in enumerate(zip(documents, similarity_scores)):
            if score >= min_similarity:
                filtered_docs.append(doc)
                filtered_metas.append(metadatas[i] if i < len(metadatas) else {})
                filtered_scores.append(score)

        if not filtered_docs:
            if verbose:
                print(
                    f"⚠️  No relevant recipes found (similarity threshold: {min_similarity:.2f})"
                )
            answer = "Mama Amaka: I couldn't find any recipes matching that in my cookbook. Try asking about jollof rice, egusi soup, or other Nigerian dishes!"
            # Add to memory
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", answer)
            return {
                "answer": answer,
                "sources": [],
                "similarity_scores": [],
                "num_chunks": 0,
            }

        if verbose:
            print(f"✓ Found {len(filtered_docs)} relevant recipe chunks")
            if filtered_scores:
                print(
                    f"  Average similarity: {sum(filtered_scores)/len(filtered_scores):.2f}"
                )

        # 2. Prepare context string from retrieved chunks
        context = ""
        sources = []
        for i, doc in enumerate(filtered_docs):
            metadata = filtered_metas[i] if i < len(filtered_metas) else {}
            source = metadata.get("source", "Unknown")
            sources.append(source)
            score_info = f" (relevance: {filtered_scores[i]:.2f})" if verbose else ""
            context += f"\n--- Recipe from {source}{score_info} ---\n{doc}\n"

        # 3. Get conversation history context
        conversation_context = self.memory.get_context()

        # 4. Generate response using the chain with conversation context (Step 7)
        try:
            answer = self.chain.invoke(
                {
                    "context": context,
                    "question": query,
                    "conversation_history": conversation_context,
                }
            )
            if verbose:
                print(f"\nMama Amaka:\n{answer}")

            # Add to memory
            self.memory.add_message("user", query, {"sources": list(set(sources))})
            self.memory.add_message(
                "assistant", answer, {"similarity_scores": filtered_scores}
            )

            return {
                "answer": answer,
                "sources": list(set(sources)),  # Unique sources
                "similarity_scores": filtered_scores,
                "num_chunks": len(filtered_docs),
            }
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            if verbose:
                print(error_msg)
            # Add to memory
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", error_msg)
            return {
                "answer": error_msg,
                "sources": [],
                "similarity_scores": [],
                "num_chunks": 0,
            }


def main():
    print("==========================================")
    print("     MAMA AMAKA - NIGERIAN RECIPE BOT     ")
    print("==========================================")

    agent = MamaAmakaAgent()

    # Ingest data on startup
    agent.ingest_data()

    print("\n------------------------------------------")
    print("Mama Amaka is ready! Ask me about Nigerian food.")
    print("Commands:")
    print("  'exit' or 'quit' - Stop the conversation")
    print("  'history' - Show conversation history")
    print("  'clear history' - Clear all conversation memory")
    print("  'export' - Export conversation to a text file")
    print("------------------------------------------")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                print("Mama Amaka: O da bo! (Goodbye!)")
                break

            if user_input.lower() == "history":
                history = agent.memory.get_last_n_exchanges(5)
                if history:
                    print("\n--- Recent Conversation ---")
                    for i, exchange in enumerate(history, 1):
                        print(f"\n({i}) You: {exchange['user']}")
                        print(f"    Mama: {exchange['assistant'][:100]}...")
                else:
                    print("No conversation history yet.")
                continue

            if user_input.lower() == "clear history":
                agent.memory.clear_history()
                print("Mama Amaka: Memory cleared. Fresh start!")
                continue

            if user_input.lower() == "export":
                filepath = agent.memory.export_conversation()
                if filepath:
                    print(f"Mama Amaka: Conversation exported to {filepath}")
                continue

            if user_input.lower() == "summary":
                print(f"Mama Amaka: {agent.memory.get_summary()}")
                continue

            if not user_input:
                continue

            result = agent.ask(user_input, verbose=True)
            # result is now a dict with 'answer', 'sources', 'similarity_scores', 'num_chunks'

        except KeyboardInterrupt:
            print("\nMama Amaka: O da bo!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
