# 
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

app = Flask(__name__)
CORS(app)

# Ensure the directory exists
data_directory = "/home/ubuntu/data"
documents = SimpleDirectoryReader("telangana").load_data()

# System prompt (no changes)
system_prompt = """
You are a Q&A assistant for Registration and Stamps Department, Government of Telangana.
Your goal is to answer questions as accurately as possible based on the instructions and context provided.

For basic conversational queries like "hi", "bye", "thank you", "ok", "no", etc., respond appropriately to make the interaction engaging and human-like. Here are some examples:
"hi": "Hello! How can I assist you today?"
"bye": "Goodbye! Have a great day!"
"thank you": "You're welcome! Is there anything else you need help with?"
"ok": "Alright, let me know if you have any more questions."
"no": "Okay, if you have any other questions later, feel free to ask."

If the question is not related to the given context, just say "Sorry, I didn't understand. Can you please provide more context or clarify your question?"
"""

# Define the query wrapper (no changes)
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Initialize LLM (adjust `device_map` based on your configuration)
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",  # Adjust based on your hardware and available GPUs
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

# Initialize embedding model (no changes)
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model,
    model_config={'protected_namespaces': ()}  # Add this line to avoid pydantic warning
)

# Initialize index (no changes)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Handle the query (no changes)
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if question:
        response = query_engine.query(question)
        return jsonify({'response': str(response)})
    else:
        return jsonify({'error': 'Invalid input'}), 400

# Handle the root URL (no changes)
@app.route('/')
def home():
    return "This is the Flask server for the Q&A assistant."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Ensure the app listens on all interfaces
