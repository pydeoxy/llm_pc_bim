# doc_handler.py
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

def load_documents(doc_paths):
    documents = []
    for path in doc_paths:
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents

def answer_document_question(question, documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    response = vectorstore.similarity_search(question)
    if response:
        return response[0].page_content
    return "Document not found for this question."
