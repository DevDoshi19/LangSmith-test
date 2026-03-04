# pip install -U langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # expects OPENAI_API_KEY in .env

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(docs)

# 3) Embed + index
# emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_documents(splits, embedding)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

config = {
    'run_name': 'pdf_rag_example',
    'metadata':{'model': 'openai/gpt-oss-20b','chain_type': 'RAG','temperature': 0.7},
    'tags': ['RAG', 'PDF', 'example', 'rag_workflow']
}

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip(), config=config)
print("\nA:", ans)
