"""
LLM Chatbot Application (Updated with OpenAI Cost Tracking + Conversation Stats)
Compatible with LangChain v0.2+
"""

import streamlit as st
from dotenv import load_dotenv
import os
import time

# LangChain Core
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")


# Streamlit page
st.set_page_config(
    page_title="LLM Chatbot Lab",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .main-header { text-align: center; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# COST TRACKING
# --------------------------

def initialize_cost_tracker():
    return {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0
    }


def update_costs(costs: dict, response_metadata: dict):
    input_tok = response_metadata.get("input_tokens", 0)
    output_tok = response_metadata.get("output_tokens", 0)
    total_tok = input_tok + output_tok
    cost_usd = response_metadata.get("cost", 0.0)

    costs["input_tokens"] += input_tok
    costs["output_tokens"] += output_tok
    costs["total_tokens"] += total_tok
    costs["total_cost_usd"] += cost_usd

    return costs


# --------------------------
# INITIALIZATION
# --------------------------

def initialize_llm(model_name: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def initialize_memory() -> BaseChatMessageHistory:
    return ChatMessageHistory()


def build_chain(system_prompt: str, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])
    return prompt | llm


# --------------------------
# PERSONAS
# --------------------------

def get_system_prompt(persona: str) -> str:
    personas = {
        "Helpful Assistant": "You are a helpful assistant.",
        "Technical Expert": "You are a precise and technical expert.",
        "Creative Writer": "You are a creative writing assistant.",
        "Socratic Teacher": "Respond by guiding through questions (Socratic style)."
    }
    return personas.get(persona, personas["Helpful Assistant"])


# --------------------------
# MAIN APP
# --------------------------

def main():

    st.markdown("<h1 class='main-header'>🤖 LLM Chatbot Lab</h1>", unsafe_allow_html=True)

    # ------------------------------------
    # SIDEBAR
    # ------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        model_name = st.selectbox(
            "Select Model",
            options=["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"],
            index=0,
        )

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

        persona = st.selectbox(
            "Chatbot Persona",
            ["Helpful Assistant", "Technical Expert", "Creative Writer", "Socratic Teacher"],
            index=0
        )

        st.divider()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = initialize_memory()
            st.session_state.costs = initialize_cost_tracker()
            st.rerun()

        st.divider()

        st.header("📊 Conversation Stats")
        if "costs" in st.session_state:
            st.metric("Total Tokens", st.session_state.costs["total_tokens"])
            st.metric("Input Tokens", st.session_state.costs["input_tokens"])
            st.metric("Output Tokens", st.session_state.costs["output_tokens"])
            st.metric("Estimated Cost (USD)", f"${st.session_state.costs['total_cost_usd']:.4f}")

        st.divider()
        st.caption("Token & cost estimates provided by OpenAI usage metadata.")

    # -------------------------
    # SESSION STATE
    # -------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()

    if "costs" not in st.session_state:
        st.session_state.costs = initialize_cost_tracker()

    # Init model + chain
    llm = initialize_llm(model_name, 
                         temperature)
    system_prompt = get_system_prompt(persona)
    chain = build_chain(system_prompt, llm)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------
    # CHAT INPUT
    # -------------------------

    if user_input := st.chat_input("Type your message here..."):

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.memory.add_user_message(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke({
                        "input": user_input,
                        "history": st.session_state.memory.messages,
                    })

                    ai_text = response.content

                    # 🎯 Update costs using OpenAI metadata
                    metadata = response.response_metadata or {}
                    st.session_state.costs = update_costs(st.session_state.costs, metadata)

                    # Display response
                    st.markdown(ai_text)

                    # Save
                    st.session_state.memory.add_ai_message(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
