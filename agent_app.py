import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# -----------------------------
# Load your fine-tuned model
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "IITKProject/iitk-companion-flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU
    return pipe

local_llm_pipeline = load_model()


# -----------------------------
# Wrap pipeline in LangChain LLM
# -----------------------------
class LocalHFLLM:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, prompt: str):
        result = self.pipeline(prompt, max_length=256, do_sample=True, top_p=0.9)
        return result[0]["generated_text"]


# -----------------------------
# Define example tools
# -----------------------------
def search_tool(query: str) -> str:
    # Dummy search tool for demonstration
    return f"Searching for: {query}"

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="useful for answering questions about general world knowledge"
    )
]


# -----------------------------
# Define prompt & agent loader
# -----------------------------
prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.

You have access to the following tools:
{tool_names}

Use the tools when needed to answer the user query.
Always reason step by step.

{agent_scratchpad}
""")

def load_agent(_tools, _llm):
    agent = create_react_agent(
        llm=_llm,
        tools=_tools,
        prompt=prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=_tools, verbose=True)
    return agent_executor


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="IITK Companion Agent", layout="wide")
st.title("🤖 IITK Companion Agent")

# Load LLM
llm = LocalHFLLM(local_llm_pipeline)

# Load agent
agent_executor = load_agent(tools, llm)

# User input
user_input = st.text_input("Ask me anything:")

if user_input:
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        st.write("### Response")
        st.write(response["output"])
