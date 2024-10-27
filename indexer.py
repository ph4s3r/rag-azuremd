import os
import time
import concurrent.futures
import openai
from termcolor import cprint
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

class Indexer:
    def __init__(self, doc_dir, chroma_path_prefix, openai_api_key, rpm=3):
        self.doc_dir = doc_dir
        self.chroma_path = f"{chroma_path_prefix}-{doc_dir[2:]}"
        self.openai_api_key = openai_api_key
        self.rpm = rpm
        self.wait_time_seconds = 60 / rpm
        openai.api_key = openai_api_key  # set the OpenAI API key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def embed_documents_with_rate_limit_handling(self, texts, retries=3):
        """attempts to embed documents and handles rate limiting with retries."""
        for attempt in range(retries):
            try:
                return self.embeddings.embed_documents(texts)
            except openai.error.RateLimitError as e:
                print(f"rate limit exceeded: {e}")
                retry_after = self.wait_time_seconds * (attempt + 1)
                print(f"retrying after {retry_after} seconds...")
                time.sleep(retry_after)
        raise Exception("rate limit error: exceeded maximum retries.")

    def load_document(self, file_path):
        """loads and splits a single markdown file into chunks"""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata = {"source": file_path.name}  # store filename as metadata
            return docs
        except Exception as e:
            print(f"error loading {file_path}: {e}")
            return []

    def index_documents(self):
        """indexes all markdown documents in the specified directory and subdirectories"""
        # check if data is already indexed
        cprint("##########################################", "blue")
        cprint("INDEXING", "green")
        if os.path.exists(self.chroma_path):
            print("data already indexed. loading existing database.")
            db_chroma = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
            return db_chroma
        else:
            cprint("no existing index found. proceeding with indexing.", "red")
        
        # load all markdown files with threading
        files = list(Path(self.doc_dir).glob("**/*.md"))
        if not files:
            cprint("no markdown files found in folder, exiting!", "red")
            return None
        print(f"found {len(files)} markdown files in the directory and subdirectories")

        # parallel document loading and processing
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.load_document, files)
            for result in results:
                all_docs.extend(result)

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(all_docs)

        # batch processing for indexing
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"processing batch {i // batch_size + 1}")

            # embed with rate limit handling
            batch_texts = [chunk.page_content for chunk in batch]
            batch_embeddings = self.embed_documents_with_rate_limit_handling(batch_texts)

            # convert chunks to Document objects with metadata
            batch_documents = [
                Document(page_content=text, metadata=chunk.metadata) for text, chunk in zip(batch_texts, batch)
            ]
            
            # add documents with metadata to Chroma
            db_chroma = Chroma.from_documents(batch_documents, self.embeddings, persist_directory=self.chroma_path)

        cprint("indexing complete.", "green")
        cprint("##########################################", "blue")
        return db_chroma
