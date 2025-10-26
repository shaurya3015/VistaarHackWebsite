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
CORS(app)  # Allow all origins

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

# Define file paths
UPLOAD_FOLDER = "uploads"
FAISS_INDEX_PATH = "my_faiss_index"
CHAT_HISTORY_FILE = "chat_history.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Embeddings Model once (This is fine, it's read-only)
print("Loading embeddings model...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embeddings model loaded.")

# Load LLM once (This is fine, it's read-only)
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
    print("PDF upload request received...")
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(pdf_path)
        print(f"PDF saved to {pdf_path}")

        # Load PDF pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        if not pages:
            print("PDF is empty")
            return jsonify({"error": "PDF is empty"}), 400

        # Split pages into chunks
        print("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)

        if not chunks:
            print("No text found in PDF")
            return jsonify({"error": "No text found in PDF"}), 400

        # Create embeddings and FAISS vector store
        print("Creating FAISS vector store...")
        # Use the global embeddings model
        vector_store = FAISS.from_documents(chunks, hf_embeddings)
        
        # Save to disk - This is the persistent state
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Vector store saved to {FAISS_INDEX_PATH}")

        # ---!! REMOVED GLOBAL VARIABLE ASSIGNMENTS !! ---
        # We no longer need to store the retriever in a global variable.
        # We will load it from the file in /ask

        # Clear chat history on new PDF upload
        try:
            os.remove(CHAT_HISTORY_FILE)
            print("Cleared old chat history.")
        except FileNotFoundError:
            pass  # No history file to clear, which is fine
        
        return jsonify({
            "message": f"PDF uploaded! Pages: {len(pages)}, Chunks: {len(chunks)}"
        })

    return jsonify({"error": "Invalid file type"}), 400


# =====================================
# API endpoint for frontend connection
# =====================================
@app.route("/ask", methods=["POST"])
def ask():
    # ---!! THIS IS THE MAIN FIX !! ---
    # 1. Check if the vector store file exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Error: FAISS index not found. PDF not uploaded.")
        return jsonify({"response": "No PDF has been uploaded yet. Please upload a PDF first."}), 400

    # 2. Load the retriever from the file
    try:
        print("Loading FAISS index from disk...")
        # We must use the *same* embeddings model to load it
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            hf_embeddings, 
            allow_dangerous_deserialization=True # Required for FAISS
        )
        pdf_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        print("Retriever loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return jsonify({"response": f"Error loading PDF knowledge base: {e}"}), 500
    # ---!! END OF FIX !! ---


    # Now the rest of your code will work, because pdf_retriever is a real object
    data = request.get_json()
    prompt = data.get("prompt")
    context_text = data.get("context") # This is context from the frontend, if any

    if not prompt:
        return jsonify({"response": "No prompt received."}), 400

    # Load chat history from file
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "rb") as f:
            chat_history = pickle.load(f)
    else:
        chat_history = []

    # Append user question
    chat_history.append({"role": "user", "content": prompt})

    # Retrieve relevant docs
    print(f"Retrieving documents for prompt: {prompt}")
    # We use the prompt *and* any context from the frontend for retrieval
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

    # Save chat history
    chat_history.append({"role": "ai", "content": ai_response})
    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump(chat_history, f)

    return jsonify({"response": ai_response})


if __name__ == "__main__":
    print("Starting Flask AI server on http://127.0.0.1:5000")
    # Port 5000 is common, but Render uses 10000. 
    # Render ignores this and uses its own port setting, so this is fine.
    app.run(port=5000, debug=True, use_reloader=False)
