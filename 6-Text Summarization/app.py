import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Streamlit APP Setup
st.set_page_config(page_title="LangChain: URL Summarizer", page_icon="🦜")
st.title("🦜 Summarize YT or Website")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter URL (YouTube or Website)")

# Model Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# 1. Define a Custom Prompt to ensure English output even for Arabic/foreign videos
summary_prompt = """
Write a concise summary of the following content in English. 
Even if the content is in another language, the final summary must be in English.
Content: {text}
CONCISE SUMMARY IN ENGLISH:"""
PROMPT = PromptTemplate(template=summary_prompt, input_variables=["text"])

if st.button("Summarize Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API Key and the URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching and processing content..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    # REMOVED: translation="en" to avoid the crash
                    # ADDED: A wider list of language codes to ensure we find a transcript
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, 
                        add_video_info=False,
                        language=["en", "ar", "fr", "es", "hi"] 
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                
                docs = loader.load()

                if not docs:
                    st.error("No content found. Ensure the video has captions enabled.")
                    st.stop()

                # 2. Text Splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(docs)

                # 3. Use custom prompts to force English output
                chain = load_summarize_chain(
                    llm, 
                    chain_type="map_reduce", 
                    map_prompt=PROMPT, 
                    combine_prompt=PROMPT,
                    verbose=False
                )
                
                output_summary = chain.invoke(split_docs)

                st.success("### Summary (Translated to English)")
                st.write(output_summary["output_text"])

        except Exception as e:
            st.error("An error occurred during processing.")
            st.exception(e)