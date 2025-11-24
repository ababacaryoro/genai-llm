# Sentiment Analysis Assignment – Comparing Classical ML, Transformers, and LLMs

## Objective
In this assignment, you will compare three different approaches to sentiment analysis using a dataset of product reviews:

1. **Classical Machine Learning classifier**
2. **Pretrained Transformer-based sentiment classifier**
3. **Large Language Model (LLM) prompting**

Your goal is to classify each review into **Positive, Neutral, or Negative** and compare the performance of the three approaches.

---

## Dataset
You are provided with a CSV file containing:

- `review_title`
- `review_comment`
- `rating` (1 to 5)

Define the target sentiment using:
- **Negative**: ratings 1–2  
- **Neutral**: rating 3  
- **Positive**: ratings 4–5  

You must push your final notebook + scripts to a **GitHub repository** and **email the repo link to: yoroba93@gmail.com**.

---

## Part 0 - Data prep

### Steps
1. Load the dataset.
2. Create a sentiment label based on the rating.
3. Take a sample of 100 rows

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("AllProductReviews.csv")

def map_rating_to_sentiment(r):
    if r in [1, 2]:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)
df["review"] = df["ReviewTitle"] + " " + df["ReviewBody"]
df_sample = df.sample(n=100, random_state=42)
```

## Part 1 — Classical Machine Learning Model

### Steps
3. Convert text into numerical features using a feature engineering template (TF-IDF for exemple).
4. Train a classifier (Logistic Regression, SVM, Random Forest, etc.).
5. Evaluate using accuracy, F1-score, and confusion matrix.

### Starter Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    df["review_comment"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

## Your code here

```

## Part 2 — Pretrained Transformer Sentiment Model

### Steps
1. Use Hugging Face transformers (e.g., distilbert-base-uncased-finetuned-sst-2-english).
2. Apply pipeline inference for each review.
3. Map the output scores ("POSITIVE"/"NEGATIVE") to the 3-class setting:

- POSITIVE → positive
- NEGATIVE → negative
- Neutral must be approximated using a threshold (e.g., |score - 0.5| < 0.1).

### Starter Code
```python

from transformers import pipeline
import pandas as pd

df = pd.read_csv("AllProductReviews.csv")

sentiment_model = pipeline("sentiment-analysis")

## Your code here

```

## Part 3 — LLM Prompting Approach
### Steps

1. Use any open LLM available (HuggingFace phi-4, Llama-3, Mistral, etc.).
2. Prompt the LLM to categorize the sentiment.
3. Ensure the output is strictly one of: positive / neutral / negative.

### Sample Prompt
````
You are a sentiment analysis model.
Classify the following product review into one of three labels: positive, neutral, or negative.
Review: "{review_text}"
Answer with only one word: positive, neutral, or negative.
````

### Starter Code (HuggingFace Inference API)
```python

classifier_flan_t5_pe_v2 = pipeline("text2text-generation", model="google/flan-t5-small")

def classify_text_flan_t5_emphasized(text):
    # Enhanced prompt to emphasize spam characteristics
    prompt = f"""You are a sentiment analysis model.
    Classify the review into one of: positive, neutral, negative. Answer with only one word.
    Review: "{text}"
    Classification:
    """
    # Remove leading/trailing whitespace and extra newlines from the prompt itself
    prompt = prompt.strip()

    result = classifier_flan_t5_pe_v2(prompt, max_new_tokens=10, num_beams=1, do_sample=False)
    prediction = result[0]['generated_text'].strip().lower()

    ## Your code here 
    ## Normalize the output


```

## Part 4 — Evaluation and Comparison

Using the ground-truth label (from the rating), compute:

- Accuracy
- Macro F1-score
- Confusion matrix
- Error analysis (show 5 examples of incorrect predictions per method)




## Deliverables

Submit a GitHub repository containing:
- ✔ Notebook with:
    - Classical ML classifier
    - Transformer-based classifier
    - LLM prompting classifier
- ✔ Additional files:
    - Scripts (if any)
    - Error analysis
- ✔ Send your repo link to:
    - 📩 yoroba93@gmail.com