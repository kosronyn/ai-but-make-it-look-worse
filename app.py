import os
import time
import streamlit as st
from google import genai
from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

openrouter_models = {
    "NVIDIA Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "NVIDIA Nemotron 3.5 Lightning": "nvidia/nemotron-3.5-lightning:free",
    "NVIDIA Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b:free",
    "LiquidAI LFM2.5 Embedding-350M": "liquid/lfm-2.5-embedding-350m:free",
    "LiquidAI LFM2.5 2.6B": "liquid/lfm-2.5-2.6b:free",
    "Z.ai GLM 5.2": "z-ai/glm-5.2:free",
    "gpt-oss-20b": "openai/gpt-oss-20b:free",
    "Poolside Languna S 2.1": "poolside/laguna-s-2.1:free",
    "Poolside Languna XS 2.1": "poolside/laguna-xs-2.1:free",
    "Google Gemma 4 26B A4B": "google/gemma-4-26b-a4b-it:free",
}

gai_studio_models = {
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
}

huggingface_models = {
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-bet",
    "Qwen3 0.6B": "Qwen/Qwen3-0.6B",
    "Qwen2.5 Coder 7B Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek V4 Flash 0731": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "Deepseek V4 Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "Z.ai GLM 4.7 Flash": "zai-org/GLM-4.7-Flash",
    "NVIDIA Nemotron 3 Nano 30B A3B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}

all_models = {**openrouter_models, **gai_studio_models, **huggingface_models}

selected_label = st.selectbox(
    "",
    options = list(all_models.keys()),
    index = None,
    placeholder = "Select a model...",
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

with st.form(key = "chat_form", clear_on_submit = True):
    user_input = st.text_input(
        "",
        placeholder = "Ask anything:"
    )
    submitted = st.form_submit_button("")

def query_provider(label, model_id, prompt):
    if label in openrouter_models:
        client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = os.getenv("OPENROUTER_API_KEY"),
        )
        response = client.chat.completions.create(
            model = model_id,
            messages = [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    elif label in gai_studio_models:
        client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model = model_id,
            contents = prompt,
        )
        return response.text

    elif label in huggingface_models:
        client = InferenceClient(api_key = os.getenv("HUGGINGFACE_API_KEY"))
        response = client.chat_completion(
            model = model_id,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 1024,
        )
        return response.choices[0].message.content

if submitted:
    if selected_label:
        if user_input:
            selected_model = all_models[selected_label]
            max_retries = 4
            retry_placeholder = st.empty()

            for attempt in range(max_retries):
                try:
                    with st.spinner("Loading..."):
                        output = query_provider(selected_label, selected_model, user_input)
                    st.write(output)
                    break

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str

                    if is_rate_limit and attempt < max_retries - 1:
                        retry_placeholder.warning(
                            f"You are being rate limited. Retrying... ({attempt + 1}/{max_retries - 1})"
                        )
                        time.sleep(3)
                        retry_placeholder.empty()
                    else:
                        st.error("The model is temporarily unavailable. Please try again later.")
                        break
        else:
            st.error("You must enter a prompt!")
    else:
        st.error("You must select a model!")