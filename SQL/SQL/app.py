from llama_cpp import Llama
import streamlit as st
from flask import Flask,request
from gevent.pywsgi import WSGIServer
import sqlite3
import pandas as pd
import os
from agents import SQL_Query_Assistant
import torch

conn = sqlite3.connect("company.db")

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    a= 100
else:
    a= 0

class LlamaCppInterface:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "../Model/Phi-3.5-mini-instruct-Q4_K_M.gguf")
        self.model_config = {
            "model_path": self.model_path,
            "n_gpu_layers": a,
            "n_ctx": 6000,
            "temperature": 0.1,
            "max_tokens":4096,
            "verbose": False
        }

    def main_llama_cpp(self, query):
        llm = Llama(**self.model_config)
        response = llm.create_chat_completion(messages=[{"role": "user", "content": query}], stream=True)
        return response

def generate_response(prompt):
    llama_interface = LlamaCppInterface()
    response = llama_interface.main_llama_cpp(prompt)
    full_response = []
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        full_response.append(chunk_message)
    return "".join(full_response)


def read_table():
    employees_df = pd.read_sql_query("SELECT * FROM Employees", conn)
    employees_table = employees_df.to_json(orient="records")

    departments_df = pd.read_sql_query("SELECT * FROM Departments", conn)
    departments_table = departments_df.to_json(orient="records")
    conn.close()
    return employees_table, departments_table



import torch
import os
import sqlite3
import pandas as pd
import streamlit as st
from llama_cpp import Llama
from agents import SQL_Query_Assistant


device = "cuda" if torch.cuda.is_available() else "cpu"
a = 100 if device == "cuda" else 0


current_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()


class LlamaCppInterface:
    def __init__(self):
        self.model_path = os.path.join(current_dir, "../Model/Phi-3.5-mini-instruct-Q4_K_M.gguf")
        self.model_config = {
            "model_path": self.model_path,
            "n_gpu_layers": a,
            "n_ctx": 6000,
            "temperature": 0.1,
            "max_tokens": 4096,
            "verbose": False
        }

    def main_llama_cpp(self, query):
        llm = Llama(**self.model_config)
        response = llm.create_chat_completion(messages=[{"role": "user", "content": query}], stream=True)
        return response


def generate_response(prompt):
    llama_interface = LlamaCppInterface()
    response = llama_interface.main_llama_cpp(prompt)

    full_response = []
    for chunk in response:
        if "choices" in chunk and chunk["choices"]:
            content = chunk["choices"][0]["delta"].get("content", "")
            full_response.append(content)

    return "".join(full_response)


def read_table():
    conn = sqlite3.connect("company.db")
    try:
        employees_df = pd.read_sql_query("SELECT * FROM Employees", conn)
        departments_df = pd.read_sql_query("SELECT * FROM Departments", conn)
        return employees_df.to_json(orient="records"), departments_df.to_json(orient="records")
    finally:
        conn.close()


# Streamlit UI
with st.sidebar:
    chat_mode = st.selectbox("Select LLM", options=["Phi-3.5-mini-instruct-Q4_K_M.gguf"])
    "[Get a Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true)"
    "[View the source code](https://github.com/Parvathy221999)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 SQL Query Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Say something")

if prompt:
    table1, table2 = read_table()
    query = SQL_Query_Assistant(table1, table2, prompt)  # Ensure this function returns a valid SQL query
    answer = generate_response(query)

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(answer)


def clear_chat_history():
    st.session_state["messages"] = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
