import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings  

# Initialize LLM (Gemini, requires API key in env)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# Initialize embeddings (HuggingFace local model, no API key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF
pdf_path = "sample.pdf"  # Replace with your PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from PDF")

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"✅ Split into {len(docs)} chunks")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)
print("✅ Created FAISS vectorstore")

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Run the QA loop
print("\n🤖 PDF QA Bot is ready! Type 'quit' to exit.\n")

while True:
    query = input("Ask a question about your PDF: ").strip()
    if not query:
        print("⚠️ Please enter a valid question (or type 'quit' to exit).")
        continue
    if query.lower() == "quit":
        print("👋 Exiting PDF QA Bot. Goodbye!")
        break

    # Run QA chain
    result = qa.invoke({"query": query})

    # Print the answer
    print("\nAnswer:")
    print(result["result"])

    # Summarize the sources (first 2 for brevity)
    print("\nSources:")
    for doc in result["source_documents"][:2]:
        page = doc.metadata.get('page', 'Unknown')
        print(f"Page {page + 1 if isinstance(page, int) else page}: {doc.page_content[:300]}...")

    print("\n" + "-"*80)
