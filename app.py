import streamlit as st
import os
import tempfile
from src.graph import graph
from src.rag import RAGSystem

st.set_page_config(page_title="LangGraph Weather & RAG Agent", layout="wide")

st.title("LangGraph Weather & RAG Agent")

# Sidebar for Setup
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload PDF for RAG", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner("Ingesting PDF..."):
            rag = RAGSystem() # Re-init to ensure clean state
            result = rag.ingest_pdf(tmp_path)
            st.success(result)
            os.remove(tmp_path)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about weather (e.g. 'Weather in London') or your PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Intermediate steps container
        steps_container = st.expander("Processing Details", expanded=False)
        
        try:
            inputs = {"question": prompt}
            
            # Stream events from the graph
            for output in graph.stream(inputs):
                for key, value in output.items():
                    with steps_container:
                        st.subheader(f"Node: {key}")
                        if key == "router":
                            st.write(f"**Decision**: {value.get('source', 'Unknown').upper()}")
                        elif key == "weather":
                            st.write("**Weather Data Fetched**")
                            st.json(value)
                        elif key == "rag":
                            st.write("**Documents Retrieved**")
                            # Truncate for display
                            context = value.get("context", "")
                            st.text(context[:500] + "..." if len(context) > 500 else context)
                        elif key == "generate":
                            full_response = value.get("answer", "")
                            # Don't display here, display in main chat
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
