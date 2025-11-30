# LAB: LLM Deployment Environment

## Building a Chatbot with Streamlit, LangChain, and OpenAI

---

## Part 1: Environment Setup

### 1.1 Create Project Directory

Open your terminal and create a new project folder:

```bash
mkdir llm-chatbot-lab
cd llm-chatbot-lab
```

### 1.2 Create Virtual Environment

It is best practice to isolate your project dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 1.3 Install Required Dependencies

Create a `requirements.txt` file with the following content:

```txt
langchain-openai
streamlit
langchain>=0.0.200
openai
python-dotenv
tiktoken
requests
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 1.4 Configure Environment Variables

Create a `.env` file to store your API key securely:

```bash
touch .env
```

Add your OpenAI API key to the `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

> **Important:** Never commit your `.env` file to version control. Add it to your `.gitignore` file.

Create a `.gitignore` file:

```gitignore
.env
venv/
__pycache__/
*.pyc
.streamlit/
```

---

## Part 2: Understanding the Architecture

Before writing code, let's understand the application architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Streamlit Frontend)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│              (LangChain Orchestration)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Prompts    │  │   Chains     │  │   Memory     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Layer                                │
│                  (OpenAI GPT API)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role | Technology |
|-----------|------|------------|
| Frontend | User interaction, message display | Streamlit |
| Orchestration | Prompt management, chain execution | LangChain |
| Memory | Conversation history management | LangChain Memory |
| Model | Text generation, understanding | OpenAI GPT |

---

## Part 3: Building the Chatbot Application

### 3.1 Create the Main Application File

Create a file named `app.py`:

```python
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

```

---

## Part 4: Advanced Features Module

### 4.1 Create a Utilities Module

Create a file named `utils.py` for reusable utility functions:

```python
"""
Utility functions for the LLM Chatbot application
"""

from typing import List, Dict, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The input text
        model: The model name for tokenizer selection
    
    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Estimate the API cost for a request.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: The model name
    
    Returns:
        Estimated cost in USD
    """
    # Pricing per 1M tokens (as of 2024)
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost


def format_conversation_history(messages: List[Dict]) -> str:
    """
    Format conversation history for display or logging.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted string representation
    """
    formatted = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        formatted.append(f"[{role}]: {content}")
    
    return "\n\n".join(formatted)


def truncate_history(messages: List[Dict], max_tokens: int = 4000, model: str = "gpt-4o-mini") -> List[Dict]:
    """
    Truncate conversation history to fit within token limits.
    Keeps the most recent messages.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens
        model: Model for tokenization
    
    Returns:
        Truncated message list
    """
    if not messages:
        return []
    
    total_tokens = 0
    truncated = []
    
    # Process messages from most recent to oldest
    for message in reversed(messages):
        msg_tokens = count_tokens(message["content"], model)
        if total_tokens + msg_tokens <= max_tokens:
            truncated.insert(0, message)
            total_tokens += msg_tokens
        else:
            break
    
    return truncated
```

### 4.2 Create a Prompt Templates Module

Create a file named `prompts.py`:

```python
"""
Prompt templates for different use cases
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


# Basic question-answer template
QA_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
)

# Summarization template
SUMMARIZATION_TEMPLATE = PromptTemplate(
    input_variables=["text", "style"],
    template="""Summarize the following text in a {style} style.

Text: {text}

Summary:"""
)

# Code explanation template
CODE_EXPLANATION_TEMPLATE = PromptTemplate(
    input_variables=["code", "language"],
    template="""Explain the following {language} code step by step.

Code:
```{language}
{code}
```

Explanation:"""
)

# Few-shot learning template for sentiment analysis
SENTIMENT_EXAMPLES = [
    {"text": "I love this product, it works perfectly!", "sentiment": "Positive"},
    {"text": "Terrible experience, would not recommend.", "sentiment": "Negative"},
    {"text": "It's okay, nothing special.", "sentiment": "Neutral"}
]

SENTIMENT_EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}"
)

SENTIMENT_TEMPLATE = FewShotPromptTemplate(
    examples=SENTIMENT_EXAMPLES,
    example_prompt=SENTIMENT_EXAMPLE_TEMPLATE,
    prefix="Classify the sentiment of the following text as Positive, Negative, or Neutral.\n\nExamples:",
    suffix="\nText: {input}\nSentiment:",
    input_variables=["input"]
)


def get_chat_prompt_template(system_message: str) -> ChatPromptTemplate:
    """
    Create a chat prompt template with a custom system message.
    
    Args:
        system_message: The system instruction
    
    Returns:
        ChatPromptTemplate instance
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
```

---

## Part 5: Running and Testing the Application

### 5.1 Launch the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### 5.2 Test Scenarios

Try these test scenarios to verify your chatbot works correctly:

**Scenario 1: Basic Conversation**
```
User: Hello! Can you introduce yourself?
Expected: The chatbot should introduce itself based on the selected persona.
```

**Scenario 2: Context Retention**
```
User: My name is Alex and I'm learning about LLMs.
User: What is my name?
Expected: The chatbot should remember and respond "Alex".
```

**Scenario 3: Temperature Effect**
```
Set temperature to 0.0, ask: "Write a one-sentence description of AI."
Set temperature to 1.0, ask the same question.
Expected: Low temperature gives consistent answers; high temperature varies.
```

**Scenario 4: Persona Switching**
```
Select "Technical Expert" and ask: "Explain how transformers work."
Select "Creative Writer" and ask the same question.
Expected: Different response styles based on persona.
```

### 5.3 Troubleshooting Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| API Key Error | Missing or invalid key | Check `.env` file and key validity |
| Import Error | Missing dependencies | Run `pip install -r requirements.txt` |
| Timeout Error | Network or API issues | Check internet connection, retry |
| Memory Error | Conversation too long | Clear conversation or restart app |

---

## Part 6: Deployment Options

### 6.1 Local Deployment (Development)

For local development and testing:

```bash
streamlit run app.py --server.port 8501
```

### 6.2 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - .:/app
    restart: unless-stopped
```

Build and run with Docker:

```bash
docker-compose up --build
```

### 6.3 Cloud Deployment (Streamlit Community Cloud)

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your `OPENAI_API_KEY` in the Secrets management section
5. Deploy

---

## Part 7: Exercise

### Add Streaming Responses

Modify the application to stream responses token by token instead of waiting for the complete response.

**Hints:**
- Use `ChatOpenAI` with `streaming=True`
- Use `st.write_stream()` in Streamlit
- Handle the callback for streaming


---


## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## Project Structure Reference

```
llm-chatbot-lab/
├── app.py                 # Main Streamlit application
├── utils.py               # Utility functions
├── prompts.py             # Prompt templates
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── README.md             # Project documentation
```

---

*Lab created for the LLM Course - Ababacar BA*
