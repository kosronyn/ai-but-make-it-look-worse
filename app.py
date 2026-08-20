import os
import time
import streamlit as st
from google import genai
from openai import OpenAI

openrouter_models = {
    "NVIDIA Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "NVIDIA Nemotron 3.5 Lightning": "nvidia/nemotron-3.5-lightning:free",
    "NVIDIA Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b:free",
    "Z.ai GLM-5.2": "z-ai/glm-5.2:free",
    "Google Gemma 4 26B A4B": "google/gemma-4-26b-a4b-it:free",
    "Google Gemma 4 31B": "google/gemma-4-31b-it:free",
    "gpt-oss-20b": "openai/gpt-oss-20b:free",
    "Poolside Languna S 2.1": "poolside/laguna-s-2.1:free",
}

gai_studio_models = {
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
}

all_models = {**openrouter_models, **gai_studio_models}

selected_label = st.selectbox(
    "",
    options = list(all_models.keys()),
    index = None,
    placeholder = "Select a model...",
    # i forgot what this placeholder is???
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

with st.form(key = "chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask anything:")
    submitted = st.form_submit_button("")

if submitted:
    if selected_label:
        if user_input:
            selected_model = all_models[selected_label]
            
            if selected_label in openrouter_models:
                client = OpenAI(
                    base_url = "https://openrouter.ai/api/v1",
                    api_key = os.getenv("OPENROUTER_API_KEY"),
                )
                max_retries = 4
                retry_placeholder = st.empty()

                for attempt in range(max_retries):
                    try:
                        with st.spinner("Loading..."):
                            response = client.chat.completions.create(
                                model = selected_model,
                                messages = [{"role": "user", "content": user_input}],
                            )

                        st.write(response.choices[0].message.content)
                        break

                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                                # max_retries - 1, because the app actually retries one less time than max_retries
                                # therefore, max_retries is increased by one during declaration to compensate
                                # this is so that it retries x amount of times instead of x - 1 amount of times
                                # before throwing an error :)
                            retry_placeholder.warning(
                                f"You are being rate limited. Retrying... ({attempt + 1}/{max_retries - 1})"
                            )
                            time.sleep(3)
                            
                            retry_placeholder.empty() 
                        else:
                            st.error("The model is temporarily unavailable. Please try again later.")
            
            elif selected_label in gai_studio_models:
                client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model = selected_model,
                    contents = user_input,
                )
                st.write(response.text)
        else:
            st.error("You must enter a prompt!")
    else:
        st.error("You must select a model!")
