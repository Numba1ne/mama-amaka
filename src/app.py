import os
import glob
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import VectorDB - use relative import since both files are in src/
from vectordb import VectorDB

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
        
        # Initialize LLM with multi-provider support
        self.llm = self._initialize_llm()
        if not self.llm:
            print("Error: No valid API key found. Please set one of:")
            print("  - OPENAI_API_KEY")
            print("  - GROQ_API_KEY")
            print("  - GOOGLE_API_KEY")
            print("in your .env file or environment variables.")
            sys.exit(1)
        
        # Create RAG prompt template (Step 6 from template)
        prompt_template_str = """You are Mama Amaka, a warm, experienced Nigerian mother and cook. 
You answer questions about Nigerian cooking using the provided recipe context.

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
                temperature=0.7  # Slightly higher for more personality
            )
        
        # Check for Groq API key
        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"  - Using Groq model: {model_name}")
            return ChatGroq(
                model=model_name,
                temperature=0.7
            )
        
        # Check for Google API key
        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"  - Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.7
            )
        
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
                        documents.append({
                            "content": content,
                            "metadata": {"source": filename}
                        })
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

    def ask(self, query: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant (Step 7 from template).
        
        Args:
            query: User's question
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            String containing the answer from the LLM
        """
        print(f"\nSearching for: '{query}'...")
        
        # 1. Retrieve relevant context chunks
        search_results = self.vector_db.search(query, n_results=n_results)
        documents = search_results.get("documents", [])
        metadatas = search_results.get("metadatas", [])
        
        if not documents:
            return "Mama Amaka: I couldn't find any recipes matching that in my cookbook. Try asking about jollof rice, egusi soup, or other Nigerian dishes!"
        
        # 2. Prepare context string from retrieved chunks
        context = ""
        for i, doc in enumerate(documents):
            # Safely get metadata with fallback
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            source = metadata.get("source", "Unknown")
            context += f"\n--- Recipe from {source} ---\n{doc}\n"
        
        # 3. Generate response using the chain (Step 7)
        try:
            answer = self.chain.invoke({"context": context, "question": query})
            print(f"\nMama Amaka:\n{answer}")
            return answer
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            print(error_msg)
            return error_msg


def main():
    print("==========================================")
    print("     MAMA AMAKA - NIGERIAN RECIPE BOT     ")
    print("==========================================")
    
    agent = MamaAmakaAgent()
    
    # Ingest data on startup
    agent.ingest_data()
    
    print("\n------------------------------------------")
    print("Mama Amaka is ready! Ask me about Nigerian food.")
    print("Type 'exit' or 'quit' to stop.")
    print("------------------------------------------")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Mama Amaka: O da bo! (Goodbye!)")
                break
            
            if not user_input:
                continue
                
            agent.ask(user_input)
            
        except KeyboardInterrupt:
            print("\nMama Amaka: O da bo!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
