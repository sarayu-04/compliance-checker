import streamlit as st
from similarity_search import vec, create_pdf_report, Synthesizer
import base64
import PyPDF2
import io
import pandas as pd
import numpy as np
from fpdf import FPDF
import zipfile
import os
import time
import shutil
import tempfile  # Import tempfile for creating temporary directories

def chunk_text(text, chunk_size=8000):
    """Split text into chunks of specified size"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def read_pdf(file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file):
    """Read text from uploaded TXT file"""
    return file.getvalue().decode("utf-8")

def process_large_text(text):
    """Process large text in chunks and combine results"""
    chunks = chunk_text(text)
    all_results = []
    
    # Process each chunk
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        # Update progress bar
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        
        # Search for each chunk
        results = vec.search(chunk, limit=3)
        all_results.append(results)
    
    # Combine results
    combined_results = pd.concat(all_results)
    
    # Convert any numpy arrays or lists to strings for deduplication
    for column in combined_results.columns:
        if hasattr(combined_results[column], 'dtype'):
            if 'object' in str(combined_results[column].dtype):
                combined_results[column] = combined_results[column].apply(
                    lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x
                )
    
    # Drop duplicates based on content
    combined_results = combined_results.drop_duplicates(subset=['content'])
    
    # Take the first 3 results if more exist
    if len(combined_results) > 3:
        combined_results = combined_results.head(3)
    
    # Create metadata dictionary
    combined_results['metadata'] = combined_results.apply(
        lambda x: {
            'agreement_date': x['agreement_date'],
            'effective_date': x['effective_date'],
            'expiration_date': x['expiration_date']
        }, 
        axis=1
    )
    
    return combined_results

def get_pdf_download_link(pdf_path):
    """Generate a download link for the PDF file"""
    with open(pdf_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="contract_analysis_report.pdf">Download PDF Report</a>'
        return href

def clean_score_format(text):
    """Clean up the compliance score format from '71-100: Excellent Compliance' to '71/100'"""
    if "Compliance Score:" in text:
        try:
            # Extract the number from the text
            score = text.split(':')[1].strip()
            if '-' in score:
                score = score.split('-')[0].strip()  # Take the first number before the dash
            if ':' in score:
                score = score.split(':')[0].strip()  # Remove any remaining text after colon
            return f"Compliance Score: {score}/100"
        except:
            return text
    return text

def create_pdf_report(response, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # Set margins to give more space for content
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Main answer section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Analysis Summary:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    # Split answer into paragraphs and clean markdown
    paragraphs = response.answer.split('\n')
    for para in paragraphs:
        # Clean the paragraph of markdown characters
        cleaned_para = para.replace('**', '').replace('*', '').strip()
        if cleaned_para:  # Only process non-empty paragraphs
            # Clean up compliance score format if present
            cleaned_para = clean_score_format(cleaned_para)
            
            # Check if it's a header
            if any(header in cleaned_para for header in ["Compliance Report:", "Strengths:", "Areas for Improvement:", "Reasoning:", "Additional Information:"]):
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(5)
                pdf.cell(0, 10, cleaned_para, ln=True)
                pdf.set_font("Helvetica", "", 11)
            else:
                # Use multi_cell to handle line breaks properly
                pdf.multi_cell(0, 7, cleaned_para)
                pdf.ln(3)
    
    # Save the PDF
    pdf.output(filename)

def create_zip_file(zip_filename, file_list):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in file_list:
            zipf.write(file, arcname=os.path.basename(file))  # Add file to zip

def main():
    st.title("Contract Analysis System")
    uploaded_files = st.file_uploader("Upload Contract Documents", type=['pdf', 'txt'], accept_multiple_files=True)
    pdf_filenames = []
    
    if uploaded_files:
        if st.button("Analyze All Contracts"):
            # Add a warning about processing time
            st.warning("Processing multiple files may take some time. Please be patient.")
            
            # Process files with rate limiting
            for uploaded_file in uploaded_files:
                try:
                    # Add processing status
                    status = st.empty()
                    status.info(f"Processing {uploaded_file.name}...")
                    
                    # Extract text based on file type
                    if uploaded_file.type == "application/pdf":
                        contract_text = read_pdf(uploaded_file)
                    else:  # txt file
                        contract_text = read_txt(uploaded_file)
                    
                    # Add delay between API calls to avoid rate limits
                    time.sleep(2)  # Increased delay to 2 seconds
                    
                    # Process the text in chunks
                    results = process_large_text(contract_text)
                    
                    # Add another delay before generating response
                    time.sleep(2)
                    
                    # Generate response with retry mechanism
                    max_retries = 3
                    retry_delay = 5  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            response = Synthesizer.generate_response(
                                question=contract_text,
                                context=results[['content', 'metadata']]
                            )
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                st.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                raise e
                    
                    # Create PDF and continue with existing processing
                    pdf_filename = f"{uploaded_file.name}_analysis_report.pdf"
                    create_pdf_report(response, pdf_filename)
                    pdf_filenames.append(pdf_filename)
                    
                    # Update status
                    status.success(f"Successfully processed {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")
                    st.info("Continuing with remaining files...")
                    continue
                
                # Add delay before processing next file
                time.sleep(3)

        # Modified zip download section
        if pdf_filenames:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create the zip file in memory using BytesIO
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for pdf_name in pdf_filenames:
                        with open(pdf_name, 'rb') as f:
                            # Write PDF to zip file
                            zip_file.writestr(os.path.basename(pdf_name), f.read())
                        # Clean up individual PDF file
                        os.remove(pdf_name)
                
                # Create download button for zip file
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Reports",
                    data=zip_buffer,
                    file_name="all_reports.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()