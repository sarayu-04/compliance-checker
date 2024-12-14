from fastapi import FastAPI
from controllers.pdf_controller import pdf_router

app = FastAPI()

# Register the PDF processing routes
app.include_router(pdf_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
