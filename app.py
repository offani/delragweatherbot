import streamlit as st
import os
import tempfile
import time
from src.graph import graph
from src.nodes import rag_system
from dotenv import load_dotenv



load_dotenv()

# Hardcoded credentials (for demonstration)
USERS = {
    "aniketh": os.getenv("pass"),
}

def login_page():
    st.markdown("# :man_technologist:  Access Portal")
    st.caption("Please login to access the Weather & RAG Agent")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form", border=True): # Use border=True for card look
            st.markdown("### User Login")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            api_key = st.text_input("Groq API Key", type="password", help="Starts with 'gsk_'", placeholder="gsk_...")
            
            st.markdown("") # Spacer
            submit = st.form_submit_button("🚀 Connect", use_container_width=True)
            
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
        st.title("⚙️ Control Panel")
        
        # Profile Section
        st.write(f"Logged in as: **{st.session_state.get('username', 'Guest')}**")
        if st.button("Log Out", icon="🚪", use_container_width=True):
            logout()
            
        st.divider()
        
        # Knowledge Base Section
        st.subheader("📂 Knowledge Base")
        
        # Initialize uploader key in session state to allow resetting
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        # File Uploader with dynamic key
        st.caption("Add new documents:")
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=["pdf"], 
            key=f"pdf_uploader_{st.session_state.uploader_key}", 
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            filename = uploaded_file.name
            
            # Use status for upload feedback
            with st.status("📄 Ingesting Document...", expanded=True) as status:
                st.write("Processing PDF...")
                
                # Save uploaded file to temp path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Ingest
                result = rag_system.ingest_pdf(tmp_path, filename)
                os.remove(tmp_path)
                
                status.update(label="✅ Upload Complete!", state="complete", expanded=False)
                
                # Increment key to reset uploader on rerun
                st.session_state.uploader_key += 1
                time.sleep(1) # Short pause to see success
                st.rerun()

        # Managed Files List
        uploaded_pdfs = rag_system.get_uploaded_pdfs()
        if uploaded_pdfs:
            st.caption(f"Managed Documents ({len(uploaded_pdfs)}):")
            for pdf_name in uploaded_pdfs:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.markdown(f"📄 **{pdf_name}**")
                with col2:
                    # Unique key for every button is crucial
                    if st.button("🗑️", key=f"btn_del_{pdf_name}", help=f"Delete {pdf_name}"):
                        result = rag_system.delete_pdf(pdf_name)
                        st.toast(f"Deleted {pdf_name}", icon="🗑️")
                        time.sleep(0.5)
                        st.rerun()
        else:
            st.info("No documents uploaded.", icon="ℹ️")
        
        st.divider()
        
        # Settings Section
        with st.expander("🛠️ Advanced Settings"):
            if st.button("🧼 Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if "thread_id" in st.session_state:
                    import uuid
                    st.session_state.thread_id = str(uuid.uuid4())
                st.toast("Conversation history cleared!", icon="🧹")
                import time
                time.sleep(0.5)
                st.rerun()

    # Chat Interface
    st.title("🌤️ Weather & RAG Agent")
    st.markdown("Ask me about the weather or upload a PDF to chat with it!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display welcome message if no history
    if not st.session_state.messages:
        st.info("👋 Welcome! I can help you with:\n- ☁️ **Weather**: 'Weather in Tokyo'\n- 📄 **Documents**: Upload a PDF and ask questions!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Use status container for dynamic feedback
            with st.status("🧠 **Thinking & Routing...**", expanded=True) as status:
                
                # Use session-based thread_id for conversation persistence
                if "thread_id" not in st.session_state:
                    import uuid
                    st.session_state.thread_id = str(uuid.uuid4())
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                inputs = {"question": prompt}
                
                final_answer = ""
                
                for output in graph.stream(inputs, config):
                    for key, value in output.items():
                        if key == "router":
                            decision = value.get('source', 'Unknown').upper()
                            status.write(f"🔀 **Decision**: {decision}")
                            if decision == "WEATHER":
                                status.update(label="🌤️ Fetching Weather Data...", state="running")
                            else:
                                status.update(label="📚 Retrieving Documents...", state="running")
                                
                        elif key == "weather":
                            status.write("✅ **Weather Data Fetched**")
                            with st.expander("View Data"):
                                st.json(value)
                        elif key == "rag":
                            status.write("✅ **Documents Retrieved**")
                            # Truncate for display
                            context = value.get("context", "")
                            with st.expander("View Context"):
                                st.text(context[:500] + "..." if len(context) > 500 else context)
                        elif key == "generate":
                            status.update(label="✨ Generating Answer...", state="running")
                            final_answer = value.get("answer", "")
                
                status.update(label="✅ **Complete**", state="complete", expanded=False)
            
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
