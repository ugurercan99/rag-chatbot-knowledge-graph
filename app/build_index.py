import os

# Disable LangSmith/LangChain tracing to prevent network-related import hangs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from dotenv import load_dotenv

print("Script started", flush=True)

# Load environment variables
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=ENV_PATH)

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'chroma_db')

def main():
    print("Inside main()", flush=True)

    print("Importing document loader...", flush=True)
    from langchain_community.document_loaders.pdf import PyPDFLoader

    print("Importing text splitter...", flush=True)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Importing embeddings...", flush=True)
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Importing Chroma...", flush=True)
    try:
        from langchain_chroma import Chroma
        chroma_backend = "langchain_chroma"
    except ImportError:
        from langchain_community.vectorstores import Chroma
        chroma_backend = "langchain_community.vectorstores"

    print("Importing torch...", flush=True)
    import torch

    # --- Load PDFs one by one, skipping unreadable files ---
    print(f"Loading documents from {DOCS_DIR}...", flush=True)

    pdf_files = sorted([
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.lower().endswith('.pdf')
    ])
    print(f"Found {len(pdf_files)} PDF files.", flush=True)

    documents = []
    skipped = []
    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        try:
            # Quick check: skip files whose data fork is empty (iCloud evicted)
            with open(pdf_path, 'rb') as f:
                header = f.read(5)
            if header != b'%PDF-':
                skipped.append(basename)
                print(f"  SKIP (not a readable PDF / iCloud evicted): {basename}", flush=True)
                continue

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  Loaded: {basename} ({len(docs)} pages)", flush=True)
        except Exception as e:
            skipped.append(basename)
            print(f"  SKIP (error): {basename}: {e}", flush=True)

    print(f"Loaded {len(documents)} document pages total.", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped}", flush=True)

    if not documents:
        print("ERROR: No documents were loaded. Check your docs/ folder.", flush=True)
        return

    print("Splitting documents into chunks...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.", flush=True)

    print("Selecting device...", flush=True)
    device = "mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    print("Loading embedding model...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 16}
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building Chroma vector store...", flush=True)
    print(f"Using Chroma backend: {chroma_backend}", flush=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=OUTPUT_DIR
    )

    print(f"Successfully built and persisted index at {OUTPUT_DIR}.", flush=True)

if __name__ == "__main__":
    main()
