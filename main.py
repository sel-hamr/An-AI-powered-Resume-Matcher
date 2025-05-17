import os
import glob
import threading
import time
import itertools
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

# Configuration
CONFIG = {
    "pdf_directory": "./resumes",
    "db_name": "resumes_db",
    "ollama_api": "http://localhost:11434/api/chat",
    "model": "deepseek-r1:7b",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retriever_k": 3,
    "temperature": 0.1,
    "max_tokens": 512,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.2
}


class ResumeProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents: List[Document] = []
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.conversation_chain = None

    def loading_animation(self, message: str, done_event: threading.Event) -> None:
        """Display a loading animation with a custom message"""
        spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
        while not done_event.is_set():
            sys.stdout.write('\r' + f"{message} {next(spinner)} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(message) + 5) + '\r')
        sys.stdout.flush()

    def start_loading(self, message: str) -> tuple:
        """Start a loading animation with the given message"""
        done_event = threading.Event()
        spinner_thread = threading.Thread(
            target=self.loading_animation,
            args=(message, done_event)
        )
        spinner_thread.daemon = True
        spinner_thread.start()
        return done_event, spinner_thread

    def load_resumes(self) -> List[Document]:
        """Load PDF resumes from the designated directory"""
        pdf_files = glob.glob(f"{self.config['pdf_directory']}/*.pdf")
        all_pages = []

        for i, file in enumerate(pdf_files):
            try:
                print(f"Loading {Path(file).name} ({i + 1}/{len(pdf_files)})")
                loader = PyPDFLoader(file)
                pages = loader.load()
                all_pages.extend(pages)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        self.documents = all_pages
        return all_pages

    def setup_embeddings(self) -> HuggingFaceEmbeddings:
        """Set up the embedding model"""
        done_event, spinner_thread = self.start_loading("Loading embedding model")

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config["embedding_model"],
                cache_folder="./embedding_cache"
            )
            self.embeddings = embeddings
        except Exception as e:
            print(f"Error setting up embeddings: {e}")
            raise
        finally:
            done_event.set()
            spinner_thread.join()

        return embeddings

    def setup_vector_store(self) -> Chroma:
        """Create or load the vector store for document retrieval"""
        if not self.documents or not self.embeddings:
            raise ValueError("Documents and embeddings must be set up first")

        done_event, spinner_thread = self.start_loading("Creating vector database")

        try:
            db_path = self.config["db_name"]
            if os.path.exists(db_path):
                Chroma(persist_directory=db_path, embedding_function=self.embeddings).delete_collection()

            vector_store = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=db_path
            )
            self.vector_store = vector_store
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            raise
        finally:
            done_event.set()
            spinner_thread.join()

        return vector_store

    def setup_llm(self) -> OllamaLLM:
        """Set up the language model"""
        llm = OllamaLLM(
            model=self.config["model"],
            api_url=self.config["ollama_api"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"],
            repetition_penalty=self.config["repetition_penalty"]
        )
        self.llm = llm
        return llm

    def setup_chain(self) -> ConversationalRetrievalChain:
        """Set up the conversation chain for resume matching"""
        if not self.vector_store or not self.llm:
            raise ValueError("Vector store and LLM must be set up first")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config["retriever_k"]}
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory
        )
        self.conversation_chain = conversation_chain
        return conversation_chain

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the conversation chain"""
        if not self.conversation_chain:
            raise ValueError("Conversation chain not set up")

        done_event, spinner_thread = self.start_loading("Ollama thinking")

        try:
            response = self.conversation_chain.invoke({"question": query})
        except Exception as e:
            print(f"Error processing query: {e}")
            raise
        finally:
            done_event.set()
            spinner_thread.join()

        return response


def get_job_details() -> Dict[str, str]:
    """Get job details from user input"""
    return {
        "job_title": input("Enter job title: "),
        "experience": input("Enter years of experience: "),
        "skills": input("Enter required skills: "),
        "education": input("Enter education requirements: ")
    }


def main():
    # Get job details
    job = get_job_details()

    # Initialize the processor
    processor = ResumeProcessor(CONFIG)

    # Set up the pipeline
    processor.load_resumes()
    processor.setup_embeddings()
    processor.setup_vector_store()
    processor.setup_llm()
    processor.setup_chain()

    # Create the query
    query = f"""
    You are a hiring manager. I will give you a job description and a resume.
    For a {job['job_title']} position requiring {job['experience']} years of experience,
    with required skills: {job['skills']}, and education: {job['education']},
    you will pick the most relevant resume from the list of resumes.
    Base your decision on how many years of experience the candidate has, what skills are relevant,
    what education is relevant, and compare companies that the candidate worked at.
    """

    # Process the query
    print("\nAnalyzing resumes against job requirements...")
    response = processor.process_query(query)

    # Display results
    print("\nResults:")
    print("-" * 80)
    print(response["answer"])
    print("-" * 80)


if __name__ == "__main__":
    main()