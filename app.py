from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# Environment Setup 
os.makedirs("/tmp/cache", exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache"
os.environ["HF_HOME"] = "/tmp/cache"

app = Flask(__name__)

#  func to load data
def load_attendance_data(csv_path):
    """Loads and formats attendance data from a CSV file into Document objects."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The CSV file was not found at the path: {csv_path}")
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Employee'], inplace=True)
    documents = []
    print(f"Loaded {len(df)} rows from CSV")

    for _, row in df.iterrows():
        # formatting
        content = (
            f"Employee {row['Employee']} (ID: {row['EmployeeID']}) "
            f"on {row['Date']}: "
            f"Status: {row['Status']}. "
            f"Time In: {row['TimeIn'] if pd.notna(row['TimeIn']) else 'N/A'}, "
            f"Time Out: {row['TimeOut'] if pd.notna(row['TimeOut']) else 'N/A'}. "
            f"Total Hours Worked: {row['Total Hours Worked'] if pd.notna(row['Total Hours Worked']) else 'N/A'}. "
            f"Leave Requested: {row['LeaveRequested'] if pd.notna(row['LeaveRequested']) else 'No'}. "
            f"Leave Type: {row['LeaveType'] if pd.notna(row['LeaveType']) else 'N/A'}. "
            f"Leave Approved: {row['LeaveApproved'] if pd.notna(row['LeaveApproved']) else 'N/A'}."
        )
        documents.append(Document(
            page_content=content,
            metadata={
                "EmployeeID": row["EmployeeID"],
                "Employee": row["Employee"],
                "Date": row["Date"],
                "Status": row["Status"],
                "LeaveRequested": row["LeaveRequested"],
                "LeaveType": row["LeaveType"],
                "LeaveApproved": row["LeaveApproved"]
            }
        ))
    return documents

CSV_PATH = "./Sample_Employees_Attendance_Data.csv"
ES_HOST = ""
ES_AUTH = ("elastic", "")
ES_INDEX_NAME = "attendance_index0045"
INFERENCE_SERVER_URL = ""
MODEL_NAME = "mistral-7b-instruct-v02"

client = Elasticsearch([ES_HOST], basic_auth=ES_AUTH, verify_certs=False)
try:
    client.info()
    print("Successfully connected to Elasticsearch.")
except Exception as e:
    print(f"Could not connect to Elasticsearch: {e}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vector_store = ElasticsearchStore(es_connection=client, index_name=ES_INDEX_NAME, embedding=embeddings)

if not client.indices.exists(index=ES_INDEX_NAME):
    print(f"Creating index '{ES_INDEX_NAME}' and loading documents...")
    try:
        raw_docs = load_attendance_data(CSV_PATH)
        print(f"Raw docs count: {len(raw_docs)}")
        
        vector_store.add_documents(raw_docs)
        client.indices.refresh(index=ES_INDEX_NAME) # Refresh index after ingestion
        print("Final indexed docs count:", client.count(index=ES_INDEX_NAME)['count'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the CSV file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred during document ingestion: {e}")
else:
    print(f"Index '{ES_INDEX_NAME}' already exists. Skipping document ingestion.")

def query_granite(prompt: str) -> str:
    """Sends a prompt to the LLM inference server and returns the response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 2072,
        "temperature": 0.01, 
        "top_p": 0.55,
        "presence_penalty": 1.00,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(INFERENCE_SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['text']
    except requests.exceptions.RequestException as e:
        raise Exception(f"Inference failed: {e}")

# last 7 days
def answer_query(query: str) -> str:
    """Handles user queries by filtering results for the last 7 days."""
    current_date = datetime(2025, 7, 10) 
    start_date = current_date - timedelta(days=6) 
    
    results = vector_store.similarity_search(query, k=50) 
    
    if not results:
        return "No relevant context found in the attendance records to answer your question."

    filtered_results = []
    for doc in results:
        try:
            doc_date_str = doc.metadata.get("Date")
            if doc_date_str:
                doc_date = datetime.strptime(doc_date_str, '%d/%m/%Y')
                if start_date <= doc_date <= current_date:
                    filtered_results.append(doc)
        except (ValueError, TypeError):
            continue

    if not filtered_results:
        return f"No relevant records were found for the period from {start_date.strftime('%d/%m/%Y')} to {current_date.strftime('%d/%m/%Y')}."

    context = "\n".join([doc.page_content for doc in filtered_results])
    print("\n--- Filtered & Retrieved Context ---")
    print(context)
    print("----------------------------------\n")

    # A more directive prompt for the LLM that includes the 7-day date range
    prompt = f"""You are an intelligent assistant that analyzes employee attendance records.
    Based *strictly and only* on the attendance records provided in the context below for the period from {start_date.strftime('%d/%m/%Y')} to {current_date.strftime('%d/%m/%Y')}, answer the user's question.
    Ensure your answer is accurate and directly supported by the context.
    If the context does not contain enough information to answer the question, state that you cannot answer based on the provided records.

    Context:
    {context}

    Question:
    {query}

    Answer:"""
    
    try:
        return query_granite(prompt)
    except Exception as e:
        return f"An error occurred during LLM inference: {str(e)}"

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests from the user and uses the RAG system."""
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"response": "Please provide a question."}), 400
    try:
        answer = answer_query(question)
        return jsonify({"answer": answer.strip()})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
