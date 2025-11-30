# Lab: LangChain & OpenAI & Hugging Face Integration



---

## Prerequisites

- Python 3.9+
- A Hugging Face account with an API token
- Knowledge of Python and LLMs

---

## Part 1: Environment Setup

### 1.1 Install Required Packages

```bash
# Core packages
pip install langchain-huggingface huggingface_hub

# For local model execution
pip install transformers accelerate bitsandbytes

# For embeddings
pip install sentence-transformers

# For OpenAI models
pip install langchain-openai

# Additional utilities
pip install langchain-core python-dotenv
```

### 1.2 Configure Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access
3. Save the token securely

```python
import os
from dotenv import load_dotenv

# Option 1: Load from .env file
load_dotenv()

# Option 2: Set directly (not recommended for production)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_token_here"
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY") # use key provided for test

# Option 3: Login interactively
from huggingface_hub import login
login()  # Will prompt for your token
```

---

## Part 2: Using Hugging Face Inference API (Serverless)

The `HuggingFacePipeline` class allows you to use models hosted on Hugging Face.

### 2.1 Basic Text Generation with HuggingFace

```python
from langchain_huggingface import HuggingFacePipeline

# Initialize the LLM using the Inference API
model = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 500},
)

# Simple invocation
response = model.invoke("What is machine learning?")
print(response)
```

The `ChatOpenAI` class allows you to use OpenAI's powerful language models.

### 2.2 Basic Text Generation with OpenAI

```python
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    max_tokens=500,
)

# Simple invocation
response = llm.invoke("What is machine learning?")
print(response.content)
```


### 2.3 Exercise: Your First LLM Call

Complete the following code to call an LLM and generate a poem about AI:

```python
from langchain_openai import ChatOpenAI

# TODO: Initialize the LLM
llm = ChatOpenAI(
    model="____",  # Choose a model: "gpt-4.1-nano", "gpt-4o", "gpt-5-nano", "gpt-4o-mini"
    temperature=____,  # Creativity level: 0.0 to 1.0
    max_tokens=____,
)

# TODO: Generate a short poem about artificial intelligence
prompt = "____"
response = llm.invoke(prompt)
print(response.content)
```

---

## Part 3: Chat Interface with ChatOpenAI

The `ChatOpenAI` class provides a conversational interface with proper message formatting.

### 3.1 Basic Chat Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the chat model
chat_model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=512,
)

# Create a conversation with system and human messages
messages = [
    SystemMessage(content="You are a helpful assistant that explains concepts simply."),
    HumanMessage(content="What is the difference between AI and Machine Learning?"),
]

# Get the response
response = chat_model.invoke(messages)
print(response.content)
```

### 3.2 Multi-turn Conversation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    max_tokens=512,
)

# Build a conversation history
conversation = [
    SystemMessage(content="You are a Python programming tutor."),
    HumanMessage(content="What is a list in Python?"),
]

# First response
response1 = chat.invoke(conversation)
print("Assistant:", response1.content)

# Add the response and continue the conversation
conversation.append(AIMessage(content=response1.content))
conversation.append(HumanMessage(content="Can you show me an example?"))

# Second response
response2 = chat.invoke(conversation)
print("Assistant:", response2.content)
```

### 3.3 Using Tuple Syntax (Simplified)

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=512,
)

# Simplified tuple syntax
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

response = chat.invoke(messages)
print(response.content)
```

### Exercise 3.1: Build a Technical Assistant

Create a chat assistant that helps users understand technical concepts:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# TODO: Initialize the chat model
chat = ChatOpenAI(
    model="____",
    temperature=____,
    max_tokens=____,
)

# TODO: Create a system message that defines the assistant's behavior
system_prompt = "____"

# TODO: Create a function that takes a user question and returns an answer
def ask_technical_question(question: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = chat.invoke(messages)
    return response.content

# Test your assistant
questions = [
    "What is Docker?",
    "Explain REST API in simple terms",
    "What is the difference between SQL and NoSQL?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask_technical_question(q)}\n")
```

---

## Part 4: Embeddings with HuggingFaceEmbeddings

Embeddings are vector representations of text, useful for semantic search and similarity. We use HuggingFace for embeddings as they provide excellent open-source models.

### 4.1 Basic Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True}
)

# Embed a single query
text = "What is machine learning?"
query_embedding = embeddings.embed_query(text)
print(f"Embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")

# Embed multiple documents
documents = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Python is a programming language.",
]
doc_embeddings = embeddings.embed_documents(documents)
print(f"Number of documents embedded: {len(doc_embeddings)}")
```

### 4.2 Semantic Similarity

```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare sentences
sentences = [
    "I love programming in Python",
    "Python is my favorite programming language",
    "The weather is nice today",
]

embeddings_list = embeddings.embed_documents(sentences)

# Calculate similarities
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings_list[i], embeddings_list[j])
        print(f"Similarity between '{sentences[i][:30]}...' and '{sentences[j][:30]}...': {sim:.4f}")
```

### Exercise 4.1: Build a Simple Semantic Search

```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Knowledge base
knowledge_base = [
    "Python is a high-level programming language known for its simplicity.",
    "JavaScript is primarily used for web development.",
    "Machine learning algorithms learn patterns from data.",
    "Docker containers package applications with their dependencies.",
    "REST APIs use HTTP methods for communication.",
    "SQL is used for managing relational databases.",
]

# TODO: Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="____"
)

# TODO: Embed all documents in the knowledge base
kb_embeddings = ____

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query: str, top_k: int = 3):
    """Find the most relevant documents for a query."""
    # TODO: Embed the query
    query_embedding = ____
    
    # TODO: Calculate cosine similarity with all documents
    similarities = []
    for i, doc_emb in enumerate(kb_embeddings):
        sim = ____  # Calculate cosine similarity
        similarities.append((i, sim))
    
    # TODO: Sort by similarity and return top_k results
    similarities.sort(key=lambda x: ____, reverse=True)
    
    results = []
    for idx, sim in similarities[:top_k]:
        results.append({
            "document": knowledge_base[idx],
            "similarity": sim
        })
    
    return results

# Test your search
queries = [
    "How do I create a web application?",
    "What language is good for beginners?",
    "How to store data in tables?",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = semantic_search(query)
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r['document'][:60]}...")
```

---

## Part 5: Building Chains with LangChain

Chains allow you to combine multiple components into a workflow.

### 5.1 Simple Prompt Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create the LLM
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=256,
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}. Provide clear and concise answers."),
    ("human", "{question}")
])

# Create the chain using the pipe operator
chain = prompt | llm | StrOutputParser()

# Invoke the chain
response = chain.invoke({
    "topic": "machine learning",
    "question": "What is overfitting?"
})
print(response)
```

### 5.2 Sequential Processing

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=512,
)

# Step 1: Generate a summary
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a summarization expert."),
    ("human", "Summarize the following text in 2 sentences:\n\n{text}")
])

# Step 2: Extract key points
keypoints_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an analyst who extracts key information."),
    ("human", "Extract 3 key points from this summary:\n\n{summary}")
])

# Build the chains
summary_chain = summary_prompt | llm | StrOutputParser()
keypoints_chain = keypoints_prompt | llm | StrOutputParser()

# Full pipeline
def analyze_text(text: str):
    summary = summary_chain.invoke({"text": text})
    keypoints = keypoints_chain.invoke({"summary": summary})
    return {
        "summary": summary,
        "keypoints": keypoints
    }

# Test
sample_text = """
Artificial Intelligence (AI) has transformed numerous industries over the past decade.
From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly
sophisticated. Machine learning, a subset of AI, enables computers to learn from data
without explicit programming. Deep learning, using neural networks with many layers,
has achieved remarkable results in image recognition, natural language processing,
and game playing. However, challenges remain, including bias in AI systems, the need
for large amounts of training data, and concerns about job displacement.
"""

result = analyze_text(sample_text)
print("Summary:", result["summary"])
print("\nKey Points:", result["keypoints"])
```

### Exercise 5.1: Build a Translation Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# TODO: Initialize the LLM
llm = ChatOpenAI(
    model="____",
    temperature=____,
    max_tokens=____,
)

# TODO: Create a translation prompt
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator."),
    ("human", """Translate the following text from {source_lang} to {target_lang}.
    
Text: {text}

Translation:""")
])

# TODO: Create a summarization prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a summarization expert."),
    ("human", "Summarize this text in one sentence:\n\n{text}")
])

# TODO: Create chains
translation_chain = ____
summary_chain = ____

def translate_and_summarize(text: str, source_lang: str, target_lang: str):
    # TODO: First translate the text
    translated = ____
    
    # TODO: Then summarize the translation
    summary = ____
    
    return {
        "translation": translated,
        "summary": summary
    }

# Test
english_text = "Artificial intelligence is changing the world in remarkable ways. It is being used in healthcare, transportation, and education to improve efficiency and outcomes."
result = translate_and_summarize(english_text, "English", "Spanish")
print("Translation:", result["translation"])
print("Summary:", result["summary"])
```
