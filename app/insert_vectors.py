import logging
from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

def truncate_text(text: str, max_bytes: int = 9900) -> str:
    """Truncate text to ensure it doesn't exceed the byte limit."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    
    # Binary search to find the right cutoff point
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        if len(text[:mid].encode('utf-8')) <= max_bytes:
            left = mid
        else:
            right = mid - 1
            
    return text[:left]

# Initialize VectorStore
vec = VectorStore()

# Delete all existing embeddings
vec.delete(delete_all=True)
logging.info("Deleted all existing embeddings")

# Read the CSV file
df = pd.read_csv("data/updated_file_with_contracts_final.csv")

# Convert date columns to datetime
date_columns = ['Agreement Date', 'Effective Date', 'Expiration Date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Prepare the data with proper metadata
processed_data = []
for _, row in df.iterrows():
    # Create metadata dictionary with only date fields
    metadata = {
        "agreement_date": row["Agreement Date"].isoformat() if pd.notna(row["Agreement Date"]) else None,
        "effective_date": row["Effective Date"].isoformat() if pd.notna(row["Effective Date"]) else None,
        "expiration_date": row["Expiration Date"].isoformat() if pd.notna(row["Expiration Date"]) else None,
    }
    
    # Combine all other columns into the content string
    content_parts = []
    for col in df.columns:
        if col != 'contract' and col not in date_columns:
            content_parts.append(f"{col}: {str(row[col]).strip()}")
    
    # Add the main contract text at the end
    if pd.notna(row['contract']):
        content_parts.append(f"Contract Text: {str(row['contract']).strip()}")
    
    # Join all content parts with newlines
    full_content = "\n".join(content_parts)
    
    # Truncate content to fit within API limits
    truncated_content = truncate_text(full_content)
    
    # Generate embedding for the truncated content
    embedding = vec.get_embedding(truncated_content)
    
    # Create record with UUID based on current time
    record = {
        "id": uuid_from_time(datetime.now()),
        "metadata": metadata,
        "contents": truncated_content,
        "embedding": embedding
    }
    processed_data.append(record)
    
    # Log progress every 10 records
    if len(processed_data) % 10 == 0:
        logging.info(f"Processed {len(processed_data)} contracts...")

# Convert to DataFrame and insert
insert_df = pd.DataFrame(processed_data)
vec.upsert(insert_df)

# Create the index
vec.create_index()

logging.info(f"Successfully inserted {len(processed_data)} contract entries")