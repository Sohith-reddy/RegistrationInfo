from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

app = Flask(_name_)
CORS(app)

# Load documents
documents = SimpleDirectoryReader("/content/drive/MyDrive/data").load_data()

# System prompt
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

# Define the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Initialize LLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

# Initialize embedding model
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Initialize index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()


# Handle the query
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if question:
        response = query_engine.query(question)
        return jsonify({'response': str(response)})
    else:
        return jsonify({'error': 'Invalid input'}), 400

# Handle the root URL
@app.route('/')
def home():
    return "This is the Flask server for the Q&A assistant."

if _name_ == '_main_':
    app.run()