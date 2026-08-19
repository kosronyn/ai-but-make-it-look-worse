import os
import streamlit as st
from google import genai

selected_model = st.selectbox(
    "",
    ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"), 
    index = None,
    placeholder = "Select a model...",
    )

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

load_css("style.css")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask anything:")
    submitted = st.form_submit_button("")

if submitted:
    if selected_model:
        if user_input:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model = selected_model,
                contents = user_input,
            )
            st.write(response.text)
        else:
            st.error("You must enter a prompt!")
    else:
        st.error("You must select a model!")