import streamlit as st
import os
import PyPDF2
import json
from sentence_transformers import SentenceTransformer, util
import openai
import tempfile

# Function to read PDF file
def read_pdf(uploaded_file):
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.seek(0)

        # Read the PDF file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(temp_file.name)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


# Function to read JSON file
def read_json(uploaded_file):
    with open(uploaded_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data['text']

# Function to get relevant context
def get_relevant_context(user_input, text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    input_embedding = model.encode(user_input)
    text_embedding = model.encode(text)
    
    # Compute cosine similarity between user input embedding and each sentence embedding
    cos_sim = util.cos_sim(input_embedding, text_embedding)
    
    # Find the index of the sentence with the highest cosine similarity
    max_index = cos_sim.argmax()
    
    # Retrieve the sentence with the highest similarity
    relevant_sentence = text[max_index]
    
    return relevant_sentence


# Function to interact with OpenAI model
def ollama_chat(user_input):
    client = openai.Client(api_key='sk-proj-5ikESwFlZUnLA3XaHfuaT3BlbkFJvFZaz7k8me9CXsOkpjgX')  # Replace 'your_api_key' with your actual API key
    response = client.completions.create(
        model="text-davinci-003",
        prompt=user_input,
        max_tokens=100
    )
    return response.choices[0].text.strip()


# Create the Streamlit app
def main():
    st.title("Document Analysis and Assistance")
    st.write("Upload a PDF or JSON file:")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "json"])

    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        if file_ext == ".pdf":
            text = read_pdf(uploaded_file)
        elif file_ext == ".json":
            text = read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return

        st.write("File Contents:")
        st.write(text)

        user_input = st.text_input("Enter your question:")
        if st.button("Get Relevant Context"):
            if user_input:
                relevant_context = get_relevant_context(user_input, text)
                st.write("Relevant Context:")
                st.write(relevant_context)
            else:
                st.warning("Please enter a question.")

        if st.button("Chat with Ollama"):
            if user_input:
                ollama_response = ollama_chat(user_input)
                st.write("Ollama's Response:")
                st.write(ollama_response)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
