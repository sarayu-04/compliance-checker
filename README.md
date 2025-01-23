
# Compliance Checker

The **Compliance Checker** is an application designed to analyze and verify contract compliance. This project utilizes PostgreSQL with pgvector to perform vector-based similarity searches, integrating AI models to analyze and compare document embeddings efficiently. PostgreSQL with pgvector offers a robust, open-source solution for managing both relational and vector data, reducing operational complexity. Key features include:
- **Efficient Vector Search**: Faster similarity queries with advanced indexing.
- **Unified Database**: Manage both structured and unstructured data in one system.
- **Open Source**: Community-driven, transparent, and cost-effective.

## Prerequisites

Before getting started, ensure you have the following installed:

- **Docker**: For containerized deployment.
- **Python 3.10**: For running the application.
- **PostgreSQL**: For storing and managing document vectors.
- **OpenAI API Key**: For generating document embeddings using OpenAI models.
- **PostgreSQL GUI Client** (optional, e.g., TablePlus): To interact with the database.

## Setup

### 1. Clone the Repository
Clone the project to your local machine:
```bash
git clone https://github.com/sarayu-04/compliance-checker.git
cd compliance-checker
```

### 2. Docker Setup

Run the following commands to set up the PostgreSQL container using Docker:
- Create a `docker-compose.yml` file with the following content:
    ```yaml
    services:
      timescaledb:
        image: timescale/timescaledb-ha:pg16
        container_name: timescaledb
        environment:
          - POSTGRES_DB=postgres
          - POSTGRES_PASSWORD=password
        ports:
          - "5432:5432"
        volumes:
          - timescaledb_data:/var/lib/postgresql/data
        restart: unless-stopped

    volumes:
      timescaledb_data:
    ```
- Start the container:
    ```bash
    docker compose up -d
    ```

### 3. Configure the Environment

Create a `.env` file by copying `example.env` and adding your OpenAI API key:
```bash
cp example.env .env
```
In `.env`, fill in your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Insert Document Vectors
Run `insert_vectors.py` to insert document chunks into the PostgreSQL database:
```bash
python app/insert_vectors.py
```
This script uses OpenAI's `text-embedding-3-small` model to generate embeddings.

### 2. Perform Similarity Search
Use `similarity_search.py` to perform similarity queries on your documents:
```bash
python app/similarity_search.py
```
This script also uses OpenAI's model to compare query embeddings against the stored document vectors.

