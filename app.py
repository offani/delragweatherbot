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
            
            # Run the graph
            # For true streaming of the final LLM response, we would need to yield tokens from the generate node. 
            # Here we visualize node execution steps.
            
            final_answer = ""
            
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
                            final_answer = value.get("answer", "")
            
            # Display final answer with typewriter effect (simulated streaming for visual)
            import time
            message_placeholder.markdown("")
            streamed_text = ""
            for char in final_answer:
                streamed_text += char
                message_placeholder.markdown(streamed_text + "▌")
                time.sleep(0.005) # adjustable speed
            message_placeholder.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
