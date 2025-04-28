
import streamlit as st
import fitz  
import os
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
import pyttsx3


DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )
    return llm

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text

def analyze_medical_report(report_text, vectorstore, hf_token):
    """Uses LLM to analyze medical data."""
    CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to identify potential medical issues from the report.
        If there are no significant issues, simply provide a general health analysis.
        Context: {context}
        Report Data: {question}
        Start the answer directly. No small talk please.
    """

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=hf_token),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    response = qa_chain.invoke({'query': report_text})
    result = response["result"]
    source_documents = response.get('source_documents', [])

    # Extract page numbers correctly
    page_numbers = [doc.metadata.get("page", "Unknown Page") for doc in source_documents]

    # Format the result output
    result_to_show = f"{result}\n\nSource document pages: {', '.join(map(str, page_numbers))}"

    # return response["result"]
    return result_to_show
def voice_input():
        """Capture voice input from the user and convert it to text."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak your query.")
            try:
                audio = recognizer.listen(source, timeout=10)
                query = recognizer.recognize_google(audio)
                st.success(f"You said: {query}")
                return query
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
            except sr.WaitTimeoutError:
                st.error("Listening timed out. Please try again.")
        return None

def text_to_speech(text):
        """Convert text to speech and read it out loud."""
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    # Conversation Mode Feature
if st.button("Enable Conversation Mode"):
        text_to_speech("I am Baymax")
        text_to_speech("your personal healtcare assistant")
        text_to_speech("how may i assist you today")
        st.info("Conversation mode enabled. Speak your query.")
        voice_query = voice_input()
        if voice_query:
            HF_TOKEN = os.environ.get("HF_TOKEN")
            HF_TOKEN = st.secrets["HF_TOKEN"]
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load vector database")
            else:
                analysis_result = analyze_medical_report(voice_query, vectorstore, HF_TOKEN)
                st.markdown(f"### Analysis Result:\n{analysis_result}")
                text_to_speech(analysis_result)

def main():
    st.title("Hi I am Baymax!")
    st.markdown("**Your personal health care assistant**")
    st.markdown("**How may I assist you today?**")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # PDF Upload Feature
    uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

    if uploaded_file:
        st.success("PDF uploaded successfully!")
        extracted_text = extract_text_from_pdf(uploaded_file)
        
        st.text_area("Extracted Report Text:", extracted_text[:1000])  # Preview first 1000 chars

        HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_TOKEN = st.secrets["HF_TOKEN"]
        vectorstore = get_vectorstore()

        if vectorstore is None:
            st.error("Failed to load vector database")
        else:
            analysis_result = analyze_medical_report(extracted_text, vectorstore, HF_TOKEN)
            st.markdown(f"### Analysis Result:\n{analysis_result}")

    # User Query Feature
    prompt = st.chat_input("Enter query:")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_TOKEN = st.secrets["HF_TOKEN"]
        vectorstore = get_vectorstore()

        if vectorstore is None:
            st.error("Failed to load vector database")
        else:
            analysis_result = analyze_medical_report(prompt, vectorstore, HF_TOKEN)
            st.chat_message('assistant').markdown(analysis_result)
            st.session_state.messages.append({'role': 'assistant', 'content': analysis_result})

if __name__ == "__main__":
    main()