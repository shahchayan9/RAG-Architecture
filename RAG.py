import os
from typing import List
import streamlit as st
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

def get_or_create_vectorstore(pdf_paths: List[Path]) -> FAISS:
    # Define embeddings directory
    working_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = Path(working_dir) / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Create a unique hash for the combination of PDFs
    combined_hash = hashlib.md5("".join(sorted([str(p) for p in pdf_paths])).encode()).hexdigest()
    vectorstore_path = embeddings_dir / combined_hash

    embeddings = OpenAIEmbeddings()
    
    if vectorstore_path.exists():
        print("Loading existing vector store.")
        return FAISS.load_local(
            str(vectorstore_path), 
            embeddings,
            allow_dangerous_deserialization=True  # Only safe because we created these files
        )
    
    # Load and process PDFs
    documents = []
    print("Loading PDFs...")
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from {pdf_path}")
        documents.extend(docs)

    # Verify document content
    # for doc in documents:
    #     print(doc.page_content[:200])  # Print the first 200 characters of each document for verification

    # Create embeddings and calculate cost
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save vectorstore
    print("Saving vector store at path:", vectorstore_path)

    vectorstore.save_local(str(vectorstore_path))
    return vectorstore

def setup_chain(vectorstore):
    if vectorstore is None:
        st.error("Failed to initialize the vector store.")
        return None

    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    
    # Fixed memory configuration with output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),  # Increase retrieval depth
        memory=memory,
        return_source_documents=True,
        chain_type="map_reduce",  # Use map_reduce for better synthesis from multiple docs
        verbose=True
    )
    
    return chain

def main():
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY in your .env file")
        return

    # Initialize session state attributes
    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        # Initialize working directory and create required folders
        working_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_dir = Path(working_dir) / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        # Paths for the two PDFs
        pdf_paths = [
            Path(working_dir) / "uploaded_pdfs" / "SOFI-2023.pdf",
            Path(working_dir) / "uploaded_pdfs" / "SOFI-2024.pdf"
        ]
        
        # Create vector store
        # print("\n"*10 + "Loading the vector store ...." + "\n"*10)
        st.session_state.vectorstore = get_or_create_vectorstore(pdf_paths)
        print("Loaded the vector store.\n\n")

    # Initialize conversation chain
    if "conversation_chain" not in st.session_state or st.session_state.conversation_chain is None:
        if st.session_state.vectorstore is not None:
            print("Setting up the conversation chain with vector store.")
            st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore)
        else:
            st.error("Vector store is not initialized, unable to set up the conversation chain.")
            return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Streamlit UI setup
    st.set_page_config(page_title="HungerLens Assistant", page_icon="🥗", layout="centered")
    st.title("HungerLens Assist 🥗")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("Ask about your PDFs...")  # For Streamlit's chat input widget

    if user_input:
        print(f"\n\n ========= User input  = {user_input} =========\n\n")
        # Custom prompt to guide the assistant toward comparison
        # query_prompt = f"Based on the 2023 and 2024 reports, provide a detailed comparison of how increased prices impacted food security in these two years. Use specific findings from each report to highlight any differences."
        with st.chat_message("user"):
            st.markdown(user_input)
        
        query_prompt = user_input
        # Check cache first
        if query_prompt in st.session_state.qa_cache:
            assistant_response = st.session_state.qa_cache[query_prompt]
            query_cost = 0
        else:
            # Get response from model
            response = st.session_state.conversation_chain({"question": query_prompt, "chat_history": st.session_state.chat_history})
            print("Conversation chain response:", response)
            source_documents = response.get("source_documents", [])
            
            # Debug: Print retrieved documents
            for doc in source_documents:
                print("Retrieved document:", doc.metadata)
            
            if source_documents:
                assistant_response = response["answer"]
                st.session_state.qa_cache[query_prompt] = assistant_response
            else:
                assistant_response = "I'm sorry, the information you requested is not available in the uploaded PDF documents. Please try a different query."
            
            # Calculate query cost (GPT-3.5-turbo: $0.0015 per 1K input tokens, $0.002 per 1K output tokens)
            input_chars = len(query_prompt) + len(str(source_documents))
            output_chars = len(assistant_response)
            input_tokens = input_chars / 4  # rough estimate
            output_tokens = output_chars / 4  # rough estimate
            query_cost = (input_tokens / 1000 * 0.0015) + (output_tokens / 1000 * 0.002)
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display new messages
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            if query_cost > 0:
                st.sidebar.write(f"Query cost: ${query_cost:.4f}")

if __name__ == "__main__":
    main()
