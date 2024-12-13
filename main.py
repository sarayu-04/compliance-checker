from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from srt import PDFProcessor  

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
        return {"message": f"File '{uploaded_file.filename}' processed successfully."}

    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code)

    except Exception as e:
        return JSONResponse(content={"detail": f"Unexpected error: {str(e)}"}, status_code=500)
