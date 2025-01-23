from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = """"""
results = vec.search(relevant_question, limit=3)

# Create a metadata dictionary from the date columns
results['metadata'] = results.apply(
    lambda x: {
        'agreement_date': x['agreement_date'],
        'effective_date': x['effective_date'],
        'expiration_date': x['expiration_date']
    }, 
    axis=1
)

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
            # Check if it's a header (like "Compliance Report:", "Strengths:", etc.)
            if any(header in cleaned_para for header in ["Compliance Report:", "Strengths:", "Areas for Improvement:", "Reasoning:", "Additional Information:"]):
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(5)
                pdf.cell(0, 10, cleaned_para, ln=True)
                pdf.set_font("Helvetica", "", 11)
            else:
                pdf.multi_cell(0, 7, cleaned_para)
                pdf.ln(3)
    
    pdf.ln(10)
    
    # Thought process section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Detailed Analysis:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    for thought in response.thought_process:
        cleaned_thought = thought.replace('**', '').replace('*', '').strip()
        if cleaned_thought:  # Only process non-empty thoughts
            if cleaned_thought.endswith(':'):
                pdf.set_font("Helvetica", "B", 11)
                pdf.ln(5)
                pdf.multi_cell(0, 7, cleaned_thought)
                pdf.set_font("Helvetica", "", 11)
            else:
                if cleaned_thought.startswith('-'):
                    pdf.multi_cell(0, 7, f"  {cleaned_thought}")
                else:
                    pdf.multi_cell(0, 7, cleaned_thought)
            pdf.ln(3)
    
    # Context information
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Context Assessment:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Sufficient context available: {response.enough_context}")
    
    # Save the PDF
    pdf.output(filename)

# Now use the correct columns
response = Synthesizer.generate_response(
    question=relevant_question, 
    context=results[['content', 'metadata']]
)

# Create PDF report
create_pdf_report(response)

# You can still keep the console output if desired
print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

# irrelevant_question = "What is the weather in Tokyo?"
# results = vec.search(irrelevant_question, limit=3)

# # Create metadata dictionary
# results['metadata'] = results.apply(
#     lambda x: {
#         'agreement_date': x['agreement_date'],
#         'effective_date': x['effective_date'],
#         'expiration_date': x['expiration_date']
#     }, 
#     axis=1
# )

# response = Synthesizer.generate_response(
#     question=irrelevant_question, 
#     context=results[['content', 'metadata']]
# )

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")

# # --------------------------------------------------------------
# # Date-based filtering
# # --------------------------------------------------------------

# # Remove or comment out the date_range filtering for now
# # date_range = {
# #     "effective_date": (datetime(2024, 1, 1), datetime(2024, 12, 31))
# # }

# results = vec.search(
#     relevant_question, 
#     limit=3
#     # Remove date_range parameter until VectorStore.search() supports it
#     # date_range=date_range  
# )

# # Create metadata dictionary
# results['metadata'] = results.apply(
#     lambda x: {
#         'agreement_date': x['agreement_date'],
#         'effective_date': x['effective_date'],
#         'expiration_date': x['expiration_date']
#     }, 
#     axis=1
# )

# response = Synthesizer.generate_response(
#     question=relevant_question, 
#     context=results[['content', 'metadata']]
# )

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")
