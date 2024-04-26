import streamlit as st
import os
import PyPDF2
import re
import json
import torch
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    if len(vault_embeddings) == 0:  # Check if the array has any elements
        return []
    # Convert the NumPy array to a PyTorch tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings_tensor)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model, ollama_model, conversation_history):
    try:
        # Get relevant context from the vault
        relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
        if relevant_context:
            # Convert list to a single string with newlines between items
            context_str = "\n".join(relevant_context)
            st.write("Context Pulled from Documents:")
            st.write(context_str)
        else:
            st.write("No relevant context found.")
        
        # Prepare the user's input by concatenating it with the relevant context
        user_input_with_context = user_input
        if relevant_context:
            user_input_with_context = context_str + "\n\n" + user_input
        
        # Append the user's input to the conversation history
        conversation_history.append({"role": "user", "content": user_input_with_context})
        
        # Create a message history including the system message and the conversation history
        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]
        
        # Send the completion request to the Ollama model
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages
        )
        
        # Append the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
        # Return the content of the response from the model
        return response.choices[0].message.content
    except Exception as e:
        st.error("An error occurred: {}".format(str(e)))
        return "An error occurred while processing your request."

# Configuration for the Ollama API client
client = openai.Client(api_key='llama3')

# Load the model and vault content
model = SentenceTransformer("all-MiniLM-L6-v2")
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
vault_embeddings = model.encode(vault_content) if vault_content else []

# Create the Streamlit app
st.title("DocuAssist: Intelligent Document Analysis and Assistance")
st.write("Welcome to DocuAssist!")
st.write("Please select a file to upload or type your question in the box below.")

# Create a file uploader
uploaded_file = st.file_uploader("Upload a PDF, JSON, or text file", type=["pdf", "json", "txt"])

# Create a text input for user input
user_input = st.text_input("Ask a question about your documents (or type 'quit' to exit): ")

# Create a dropdown menu for selecting the Ollama model
model_select = st.selectbox("Select an Ollama model:", ["llama3", "babbage", "curie", "davinci"])

# Create a button to trigger the Ollama chat
if st.button("Chat with Ollama"):
    response = ollama_chat(user_input, "You are a helpful assistant that is an expert at extracting the most useful information from a given text", vault_embeddings, vault_content, model, model_select, [])
    st.write("Response:")
    st.write(response)

# Create a dropdown menu for selecting the file type
file_type = st.selectbox("Select a file type:", ["PDF", "JSON", "Text"])

# Create a button to upload a file
if st.button("Upload File"):
    if uploaded_file:
        if file_type == "PDF":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with open("temp.pdf", 'rb') as f:
                pdf_reader = PyPDF2.PdfFileReader(f)
                num_pages = len(pdf_reader.pages)
                text = ''
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    if page.extract_text():
                        text += page.extract_text() + " "
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(text.strip() + "\n\n")
        elif file_type == "JSON":
            with open("temp.json", 'w', encoding="utf-8") as f:
                f.write(uploaded_file.getvalue().decode("utf-8"))
            with open("temp.json", 'r', encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(text.strip() + "\n\n")
        elif file_type == "Text":
            with open("temp.txt", 'w', encoding="utf-8") as f:
                f.write(uploaded_file.getvalue().decode("utf-8"))
            with open("temp.txt", 'r', encoding="utf-8") as f:
                text = f.read()
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(text.strip() + "\n\n")

# Create a button to exit the app
if st.button("Exit"):
    st.write("Goodbye!")
    st.stop()
