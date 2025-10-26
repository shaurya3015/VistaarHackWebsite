# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from quiz_backend import register_quiz_routes

# ===============================
# Load everything once on startup
# ===============================

app = Flask(__name__)
register_quiz_routes(app)
CORS(app)

# Your PDF file path

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# ===============================
# Upload folder & globals
# ===============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pdf_pages = []          # Loaded pages from uploaded PDF
pdf_vector_store = None # FAISS vector store accessible globally
pdf_retriever = None

# ===============================
# PDF upload route
# ===============================
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    print("PDF found")
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(pdf_path)

        # Load PDF pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        if not pages:
            return jsonify({"error": "PDF is empty"}), 400

        # Split pages into chunks and pickle
        print("PDF_found_2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        if not chunks:
            return jsonify({"error": "No text found in PDF"}), 400

        # Create embeddings and FAISS vector store
        hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = FAISS.from_documents(chunks, hf)
        vector_store.save_local("my_faiss_index")
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        # Store globally so chatbot can access
        global pdf_pages, pdf_vector_store, pdf_retriever
        pdf_pages = pages
        pdf_vector_store = vector_store
        pdf_retriever = retriever
        #print("PDF found + retriever made")
        print("PDF found + retriever made")
        try:
            os.remove("chat_history.pkl")

        except FileNotFoundError:
            pass

        return jsonify({
            "message": f"PDF uploaded! Pages: {len(pages)}, Chunks: {len(chunks)}"
        })

    return jsonify({"error": "Invalid file type"}), 400

# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=150,
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

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

# Load chat history
chat_file = "chat_history.pkl"
if os.path.exists(chat_file):
    with open(chat_file, "rb") as f:
        chat_history = pickle.load(f)
else:
    chat_history = []


# =====================================
# API endpoint for frontend connection
# =====================================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt")
    context_text = data.get("context")

    if not prompt:
        return jsonify({"response": "No prompt received."}), 400

    # Append user question
    chat_history.append({"role": "user", "content": prompt+str(context_text)})

    # Retrieve relevant docs
    retrieved_docs = pdf_retriever.invoke(prompt+str(context_text))
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
    result = model.invoke(final_prompt)
    ai_response = result.content if hasattr(result, "content") else str(result)

    # Save chat history
    chat_history.append({"role": "ai", "content": ai_response})
    with open(chat_file, "wb") as f:
        pickle.dump(chat_history, f)

    return jsonify({"response": ai_response})


if __name__ == "__main__":
    print("Starting Flask AI server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True, use_reloader=False)


