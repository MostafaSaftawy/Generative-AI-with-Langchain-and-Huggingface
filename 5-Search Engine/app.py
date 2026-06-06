import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os

# 1. Tools Setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun()

tools = [search, arxiv, wiki]

st.title("🔎 LangChain - Chat with Search (2026 Edition)")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    if not api_key:
        st.info("Please add your Groq API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the modern model
    llm = ChatGroq(groq_api_key=api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    # Create the modern LangGraph-based agent
    # We pass the model and the tools directly
    agent_executor = create_react_agent(llm, tools)

    with st.chat_message("assistant"):
        # We use st.status to show the agent's "thinking" process in 2026
        with st.status("Searching and reasoning...", expanded=True) as status:
            # We pass the prompt as a HumanMessage
            inputs = {"messages": [HumanMessage(content=prompt)]}
            response = agent_executor.invoke(inputs)
            
            # The last message in the response is the agent's final answer
            final_answer = response["messages"][-1].content
            status.update(label="Research complete!", state="complete", expanded=False)

        st.write(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})