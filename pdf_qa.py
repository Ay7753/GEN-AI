import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
  

# =========================
# 1. Initialize Models
# =========================
# LLM (Gemini) -> requires GOOGLE_API_KEY in environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# Embeddings (HuggingFace local model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# =========================
# 2. Load PDF
# =========================
pdf_path = "sample.pdf"  # Replace with your PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from PDF")

# =========================
# 3. Split into Chunks
# =========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"✅ Split into {len(docs)} chunks")

# =========================
# 4. Create Embeddings for ALL Chunks
# =========================
chunk_texts = [d.page_content for d in docs]
chunk_embeddings = embeddings.embed_documents(chunk_texts)

# Save chunks + embeddings into a DataFrame
df = pd.DataFrame({
    "chunk_id": range(1, len(docs) + 1),
    "page_number": [d.metadata.get("page", "Unknown") for d in docs],
    "chunk_text": chunk_texts,
    "embedding": chunk_embeddings
})

# Save to CSV
output_csv = "chunks_with_embeddings.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(docs)} chunks + embeddings to {output_csv}")

# =========================
# 5. Build Vectorstore
# =========================
vectorstore = FAISS.from_documents(docs, embeddings)
print("✅ Created FAISS vectorstore")

# =========================
# 6. Create QA Chain
# =========================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# =========================
# 7. Run QA Loop
# =========================
print("\n🤖 PDF QA Bot is ready! Type 'quit' to exit.\n")

while True:
    query = input("Ask a question about your PDF: ").strip()
    if not query:
        print("⚠️ Please enter a valid question (or type 'quit' to exit).")
        continue
    if query.lower() == "quit":
        print("👋 Exiting PDF QA Bot. Goodbye!")
        break

    # Run QA
    result = qa.invoke({"query": query})

    # Print answer
    print("\nAnswer:")
    print(result["result"])

    # Print sources
    print("\nSources:")
    for doc in result["source_documents"][:2]:
        page = doc.metadata.get('page', 'Unknown')
        print(f"Page {page + 1 if isinstance(page, int) else page}: {doc.page_content[:300]}...")

    print("\n" + "-"*80)
