# app.py

import streamlit as st
from rag_engine import get_rag_chain

# Set the page configuration for your app
st.set_page_config(page_title="IITK Companion", page_icon="📚")

# --- App Title and Description ---
st.title("📚 IITK Companion")
st.write(
    "Welcome to the IITK Companion! Ask any question about the Aerospace Engineering "
    "department courses or the general UG Manual regulations."
)

# --- Initialize the RAG Chain ---
# We use st.cache_resource to load the model only once, so it's fast after the first run.
@st.cache_resource
def load_chain():
    """Loads the RAG chain and caches it for the Streamlit session."""
    chain = get_rag_chain()
    return chain

# Load the chain, showing a spinner during the initial load
with st.spinner("Loading the AI model, please wait..."):
    chain = load_chain()

# --- User Input and Response ---
st.header("Ask a Question")
user_question = st.text_input("e.g., What is the description for AE201A?")

# Process the question when the user enters something
if user_question:
    # Show a spinner while the AI is thinking
    with st.spinner("Searching for the answer..."):
        try:
            # Get the result from the chain using the .invoke() method
            result = chain.invoke(user_question)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(result["result"])
            
            # (Optional) Display source documents for verification
            with st.expander("Show Sources"):
                st.write("These are the sections from the documents used to generate the answer:")
                for document in result["source_documents"]:
                    st.write("---")
                    st.write(document.page_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")