from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.pdf_processor import PDFProcessor
from config.settings import UPLOAD_DIR

pdf_router = APIRouter()
pdf_processor = PDFProcessor(upload_dir=UPLOAD_DIR)

@pdf_router.post("/upload-pdf/")
async def upload_pdf(uploaded_file: UploadFile = File(...)):
    try:
        # Process the uploaded PDF file
        file_path = pdf_processor.load_pdf(uploaded_file)
        docs = pdf_processor.extract_text(file_path)
        chunks = pdf_processor.chunk_text(docs)
        embeddings = pdf_processor.create_embeddings(chunks)
        pdf_processor.store_embeddings_local(embeddings, chunks)

        return {"message": f"File '{uploaded_file.filename}' processed successfully."}

    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code)

    except Exception as e:
        return JSONResponse(content={"detail": f"Unexpected error: {str(e)}"}, status_code=500)
