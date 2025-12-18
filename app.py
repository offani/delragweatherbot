import streamlit as st
import os
import tempfile
from src.graph import graph
from src.nodes import rag_system
from dotenv import load_dotenv



load_dotenv()

# Hardcoded credentials (for demonstration)
USERS = {
    "admin": "password123",
    "user": "pass",
    "test": "test"
}

def login_page():
    st.title("🔐 Login to Weather & RAG Agent")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key starting with 'gsk_'")
        
        submit = st.form_submit_button("Login")
        
        if submit:
            if username in USERS and USERS[username] == password:
                if api_key and api_key.startswith("gsk_"):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    os.environ["GROQ_API_KEY"] = api_key
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Please enter a valid Groq API Key (starts with 'gsk_').")
            else:
                st.error("Invalid username or password")

def logout():
    files = rag_system.get_uploaded_pdfs()
    for file in files:
        rag_system.delete_pdf(file)
    st.session_state.clear()
    # explicitly clear env var if desired, though os.environ persists for the process
    # os.environ.pop("GROQ_API_KEY", None)
    st.rerun()

# Check authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    # --- Main Application Code ---
    
    # Sidebar for Setup
    with st.sidebar:
        st.header(f"User: {st.session_state.get('username', 'Unknown')}")
        if st.button("🚪 Logout"):
            logout()
            
        st.divider()
        st.header("Setup")
        
        # Clear conversation history button
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.messages = []
            if "thread_id" in st.session_state:
                import uuid
                st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        
        st.divider()
        
        # Show uploaded PDFs
        st.subheader("📚 Uploaded PDFs")
        uploaded_pdfs = rag_system.get_uploaded_pdfs()
        
        if uploaded_pdfs:
            for pdf_name in uploaded_pdfs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {pdf_name}")
                with col2:
                    if st.button("🗑️", key=f"delete_{pdf_name}"):
                        result = rag_system.delete_pdf(pdf_name)
                        st.success(result)
                        st.rerun()
        else:
            st.info("No PDFs uploaded yet")
        
        st.divider()
        
        uploaded_file = st.file_uploader("Upload PDF for RAG", type=["pdf"], key="pdf_uploader")
        if uploaded_file:
            filename = uploaded_file.name
            
            # Track if this file was already processed
            if "last_uploaded_file" not in st.session_state:
                st.session_state.last_uploaded_file = None
            
            # Only process if it's a new file
            if st.session_state.last_uploaded_file != filename:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                with st.spinner("Ingesting PDF..."):
                    result = rag_system.ingest_pdf(tmp_path, filename)
                    st.success(result)
                    os.remove(tmp_path)
                    st.session_state.last_uploaded_file = filename

    # Chat Interface
    st.title("LangGraph Weather & RAG Agent")
    
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
            
            # try:
            # Use session-based thread_id for conversation persistence
            if "thread_id" not in st.session_state:
                import uuid
                st.session_state.thread_id = str(uuid.uuid4())
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            inputs = {"question": prompt}
            
            # Run the graph with checkpointer
            # For true streaming of the final LLM response, we would need to yield tokens from the generate node. 
            # Here we visualize node execution steps.
            
            final_answer = ""
            
            for output in graph.stream(inputs, config):
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
                

            # except Exception as e:
            #     st.error(f"An error occurred: {e}")
