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