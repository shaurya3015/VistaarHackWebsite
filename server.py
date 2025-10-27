from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from quiz_backend import register_quiz_routes

# ===============================
# App Initialization
# ===============================

app = Flask(__name__)
register_quiz_routes(app)
#CORS(app)  
CORS(app, resources={
    r"/ask": {"origins": "*"},
    r"/upload_pdf": {"origins": "*"}
    # If you are using quiz_backend.py, add its routes here too
    # e.g., r"/quiz/*": {"origins": "*"}
})# Allow all origins

# ===============================
# CRITICAL: SET YOUR API TOKEN IN RENDER'S ENVIRONMENT VARIABLES
# DO NOT hardcode it here.
# The code will automatically read it from the environment.
# ===============================
    # You might want to raise an exception here or set a default for local testing
    # For Render, it *must* be in the environment variables.

# ===============================
# Global Read-Only Objects & Config
# ===============================
pdf_retriever = None
chat_history = []

# ===============================
# Global Read-Only Objects (Loaded once at startup)
# ===============================
print("Loading embeddings model...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embeddings model loaded.")

print("Loading LLM...")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=150,
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
print("LLM loaded.")

# Define prompt template
prompt_template = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided context and chat_history.
If the question is not answerable from context, say "I don't know."
Keep answers concise (max 20 words).
Context:
{context}
Chat history:
{chat_history_text}
Question: {question}
""",
    input_variables=["context", "question", "chat_history_text"],
)


# ===============================
# PDF upload route
# ===============================
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    # We must modify the global variables
    global pdf_retriever, chat_history
    
    print("PDF upload request received...")
    if "pdf_file" not in request.files:
        print("Error: No file part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        print("Error: No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        # We must save the file temporarily to give PyPDFLoader a path
        UPLOAD_FOLDER = "uploads" # This folder is temporary and will be wiped
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        try:
            file.save(pdf_path)
            print(f"PDF temporarily saved to {pdf_path}")

            # Load PDF pages
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            if not pages:
                print("Error: PDF is empty or could not be read.")
                return jsonify({"error": "PDF is empty"}), 400

            # Split pages into chunks
            print("Splitting documents...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)

            if not chunks:
                print("Error: No text found in PDF after splitting.")
                return jsonify({"error": "No text found in PDF"}), 400

            # Create embeddings and FAISS vector store IN MEMORY
            print("Creating FAISS vector store in memory...")
            vector_store = FAISS.from_documents(chunks, hf_embeddings)
            
            # ---!! THIS IS THE CORE LOGIC !! ---
            # 1. Store the retriever in the global variable (in RAM)
            pdf_retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 3}
            )
            
            # 2. Clear the global chat history for the new session
            chat_history = []
            
            print("Vector store created in RAM and ready.")
            
            return jsonify({
                "message": f"PDF processed! Pages: {len(pages)}, Chunks: {len(chunks)}"
            })

        except Exception as e:
            print(f"An error occurred during PDF processing: {e}")
            return jsonify({"error": f"Server error: {e}"}), 500
        
        finally:
            # 3. Clean up the temporary file
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    print(f"Removed temp file {pdf_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")

    print("Error: Invalid file type.")
    return jsonify({"error": "Invalid file type, must be .pdf"}), 400


# =====================================
# API endpoint for frontend connection
# =====================================
@app.route("/ask", methods=["POST"])
def ask():
    # We will READ from the global variables
    global pdf_retriever, chat_history
    
    # 1. Check if the retriever exists in memory
    if pdf_retriever is None:
        print("Error: PDF retriever not found in memory. User needs to upload.")
        return jsonify({"response": "No PDF has been uploaded yet. Please upload a PDF first."}), 400

    # --- Everything below assumes pdf_retriever exists ---
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        context_text = data.get("context") 

        if not prompt:
            print("Error: No prompt received in /ask request.")
            return jsonify({"response": "No prompt received."}), 400

        # chat_history is already loaded from the global variable
        chat_history.append({"role": "user", "content": prompt})

        # Retrieve relevant docs
        print(f"Retrieving documents for prompt: {prompt}")
        retrieval_query = prompt + str(context_text or "") 
        retrieved_docs = pdf_retriever.invoke(retrieval_query)
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Merge with frontend-sent context (optional)
        full_context = (context_text or "") + "\n\n" + retrieved_context

        # Combine chat history
        chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

        # Create final prompt
        final_prompt = prompt_template.invoke({
            "context": full_context,
            "question": prompt,
            "chat_history_text": chat_history_text
        })

        # Run model
        print("Invoking model...")
        result = model.invoke(final_prompt)
        ai_response = result.content if hasattr(result, "content") else str(result)
        print(f"Model response: {ai_response}")

        # Save chat history to the global variable
        chat_history.append({"role": "ai", "content": ai_response})
        
        return jsonify({"response": ai_response})

    except Exception as e:
        print(f"An error occurred during /ask: {e}")
        return jsonify({"response": f"An error occurred: {e}"}), 500


if __name__ == "__main__":
    print("Starting Flask AI server in debug mode on http://127.0.0.1:5000")
    # Port 5000 is fine for local. Render/Gunicorn will ignore this.
    app.run(port=5000, debug=True)


