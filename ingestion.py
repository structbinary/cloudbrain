import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

class TerraformIngestion:
    def __init__(self, vector_store_path: str = "./data/chroma"):
        """Initialize the Terraform ingestion process.
        
        Args:
            vector_store_path: Path to store the vector database
        """
        self.vector_store_path = vector_store_path
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings = OpenAIEmbeddings()
        self.collection_name = "terraform_docs" 
        
    def read_repository_urls(self) -> List[str]:
        """Read repository URLs from the terraform-repository.url file."""
        with open("terraform-repository.url", "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    def clone_repository(self, repo_url: str) -> str:
        """Clone a Git repository to a temporary directory.
        
        Args:
            repo_url: URL of the Git repository
            
        Returns:
            Path to the cloned repository
        """
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(self.temp_dir, repo_name)
        
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print(f"Cloned {repo_url} to {repo_path}")
        
        return repo_path
    
    def find_terraform_files(self, repo_path: str) -> List[str]:
        """Find all Terraform and YAML files in the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of file paths
        """
        terraform_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".tf", ".tfvars", ".yaml", ".yml")):
                    terraform_files.append(os.path.join(root, file))
        
        return terraform_files
    
    def process_terraform_files(self, repo_path: str) -> List[Dict[str, Any]]:
        """Process Terraform files and extract relevant information.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of processed documents
        """
        terraform_files = self.find_terraform_files(repo_path)
        documents = []
        
        for file_path in terraform_files:
            with open(file_path, "r") as f:
                content = f.read()
                
            # Create a document with metadata
            relative_path = os.path.relpath(file_path, repo_path)
            repo_name = os.path.basename(repo_path)
            
            documents.append({
                "content": content,
                "metadata": {
                    "source": file_path,
                    "repo": repo_name,
                    "relative_path": relative_path,
                    "file_type": os.path.splitext(file_path)[1]
                }
            })
        
        return documents
    
    def load_documents_to_vectorstore(self, documents: List[Dict[str, Any]]):
        """Load documents into the vector store.
        
        Args:
            documents: List of documents with content and metadata
        """
        # Create text documents for LangChain
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(langchain_docs)
        
        # Create or update the vector store with specific collection name
        print(f"DEBUG: Creating vector store at {self.vector_store_path} with collection {self.collection_name}")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path,
            collection_name=self.collection_name
        )
        
        print(f"Loaded {len(splits)} document chunks into vector store at {self.vector_store_path}")
    
    def run(self):
        """Run the complete ingestion process."""
        repo_urls = self.read_repository_urls()
        print(f"Found {len(repo_urls)} repositories to process")
        
        all_documents = []
        
        for repo_url in repo_urls:
            try:
                repo_path = self.clone_repository(repo_url)
                documents = self.process_terraform_files(repo_path)
                all_documents.extend(documents)
                print(f"Processed {len(documents)} files from {repo_url}")
            except Exception as e:
                print(f"Error processing {repo_url}: {e}")
        
        self.load_documents_to_vectorstore(all_documents)
        print(f"Total documents processed: {len(all_documents)}")

if __name__ == "__main__":
    ingestion = TerraformIngestion()
    ingestion.run()
