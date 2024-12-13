import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import pickle

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PDFProcessor:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        self.embedding_file = "embeddings.pkl"
    
    def load_pdf(self, uploaded_file: UploadFile):
        try:
            if not uploaded_file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
            
            file_path = os.path.join(self.upload_dir, uploaded_file.filename)
            with open(file_path, "wb") as f:
                content = uploaded_file.file.read()
                f.write(content)

            return file_path

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading PDF: {str(e)}")

    def extract_text(self, file_path: str):
        try:
            docs = PyPDFLoader(file_path).load()
            return docs
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

    def chunk_text(self, docs):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            return chunks
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error chunking text: {str(e)}")

    def create_embeddings(self, chunks):
        try:
            hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            embeddings = [hf_embeddings.embed_documents([chunk.page_content])[0] for chunk in chunks]
            return embeddings
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

    def store_embeddings_local(self, embeddings, chunks):
        try:
            # Create FAISS index with embeddings and metadata (chunks)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            with open(self.embedding_file, "wb") as f:
                pickle.dump(vectorstore, f)
            return vectorstore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing embeddings locally: {str(e)}")

    def connect_to_server(self, embeddings):
        pass

app = FastAPI()
pdf_processor = PDFProcessor(upload_dir=UPLOAD_DIR)

@app.post("/upload-pdf/")
async def upload_pdf(uploaded_file: UploadFile = File(...)):
    try:
        file_path = pdf_processor.load_pdf(uploaded_file)
        docs = pdf_processor.extract_text(file_path)
        chunks = pdf_processor.chunk_text(docs)
        embeddings = pdf_processor.create_embeddings(chunks)
        vectorstore = pdf_processor.store_embeddings_local(embeddings, chunks)
        pdf_processor.connect_to_server(embeddings)
        return {"message": f"File '{uploaded_file.filename}' processed successfully."}

    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code)

    except Exception as e:
        return JSONResponse(content={"detail": f"Unexpected error: {str(e)}"}, status_code=500)
