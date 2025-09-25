from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
# Not using langchain_chroma as per discussion here: https://github.com/langchain-ai/langchain/discussions/20449
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import tiktoken
from typing import Optional


def build_vectorstore(doc_folder: str, documents: Optional[int] = 5, rebuild: bool = True) -> Chroma:

    """
    This function builds a vectorstore by chunking pdf files in a supplied directory

    Args:
        doc_folder (str): directory containing pdf files to chunk
        documents (int): the total number of documents to use. Pass None to use all documents.
        rebuild (bool): Whether to rebuild the database or open an existing database

    Returns:
        db (Chroma): a Chroma vectorstore

    """
    vectorstore_dir = 'chroma_db'

    if rebuild == False:

        if not vectorstore_dir in os.listdir():
            raise FileNotFoundError('No database found - rebuild required.')
        
        db = Chroma(persist_directory=f"./{vectorstore_dir}", embedding_function=OpenAIEmbeddings())   

    else:
        os.rmdir(vectorstore_dir)
        
        docs = os.listdir(doc_folder)

        # Reduce number of documents if desired
        if documents is not None:
            documents = min(documents, len(docs))           
            docs = docs[-documents:]

        text_splitter = RecursiveCharacterTextSplitter(separators = ['\n \n']
                                                    , chunk_size=1000
                                                    , length_function=len
                                                    , chunk_overlap=100)

        documents = []

        for doc in docs:    
            pdf_path = os.path.join(doc_folder, doc)
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())

        ## Update document metadata
        for doc in documents:
            doc.metadata['year'] = int(re.search(r'\d+', doc.metadata['source']).group(0))

        doc_chunks = text_splitter.split_documents(documents)    

        # create the open-source embedding function
        embedding_function = OpenAIEmbeddings()

        # load it into Chroma and save locally
        db = Chroma.from_documents(doc_chunks, embedding_function, persist_directory=f"./{vectorstore_dir}")

    return db


## Function to filter document context to ensure it doesn't exceed {token_limit} tokens
def filter_context(context: list[Document], model_name: str, token_limit: int) -> list[Document]:

    """
    This function accepts a list of documents that serve as context for a RAG application, and filters them
    to ensure the total context does not exceed a given token limit

    Args:
        context (list): a list of Documents
        model_name (str): LLM in use (to identify the appropriate encoding)
        token_limit (int): The upper limit of context tokns

    Returns:
        filtered_context (list): a list of Documents with total tokens less than the specified token limit

    """

    # Create the approriate tokeniser to count tokens
    encoding = tiktoken.encoding_for_model(model_name)

    # List to hold filtered documents
    filtered_context = []
    tokens = 0 # Counter
    
    # Only add document if it doesn't exceed the token limit
    for d in context:
        tokens += len(encoding.encode(d.page_content))
        if tokens < token_limit:
            filtered_context += [d]

    return filtered_context