"""
Utility functions for the LLM Fashion RAG Chatbot application
Includes: Token counting, cost estimation, RAG preparation, and guardrails
"""

from typing import List, Dict, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage 
from langchain_core.documents import Document
import tiktoken
import pandas as pd
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# =============================================================================
# TOKEN COUNTING & COST ESTIMATION
# =============================================================================

def count_tokens(text: str, model: str = "gpt-5-nano") -> int:
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
        "gpt-5-nano": {"input": 0.05, "output": 0.4},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.4},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-5-nano"])
    
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost


def load_vectorstore(persist_directory: str, collection_name: str):
    """
    Charge une base vectorielle existante depuis le disque.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print('collection_name:', collection_name)  # Debug log
    print(f"Persistent dir {os.listdir(persist_directory)}")  # Debug log to check files in directory
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    collection_count = vectorstore._collection.count()
    
    return vectorstore, f"✅ {collection_count} produits chargés"

# =============================================================================
# CONVERSATION FORMATTING
# =============================================================================

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


def truncate_history(messages: List[Dict], max_tokens: int = 4000, model: str = "gpt-5-nano") -> List[Dict]:
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


# =============================================================================
# RAG DATA PREPARATION
# =============================================================================

def clean_fashion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataframe fashion pour la préparation RAG.
    
    Args:
        df: DataFrame brut (fusion de styles et images)
    
    Returns:
        DataFrame nettoyé
    """
    df_clean = df.copy()
    
    # Supprimer les lignes sans ID ou sans nom de produit
    df_clean = df_clean.dropna(subset=['id', 'productDisplayName'])
    
    # Remplir les valeurs manquantes
    fill_values = {
        'gender': 'Unisex',
        'masterCategory': 'Unknown',
        'subCategory': 'Unknown',
        'articleType': 'Unknown',
        'baseColour': 'Unknown',
        'season': 'All Season',
        'year': 'Unknown',
        'usage': 'Casual',
        'link': ''
    }
    df_clean = df_clean.fillna(fill_values)
    
    # Convertir l'ID en int
    df_clean['id'] = df_clean['id'].astype(int)
    
    # Nettoyer les espaces
    string_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 
                      'baseColour', 'season', 'usage', 'productDisplayName']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean


def create_product_description(row: pd.Series) -> str:
    """
    Crée une description textuelle complète d'un produit pour l'embedding.
    Cette description sera utilisée pour la recherche sémantique.
    
    Args:
        row: Ligne du DataFrame représentant un produit
    
    Returns:
        Description textuelle enrichie
    """
    # Extraction de la marque (premier mot du productDisplayName)
    product_name = str(row['productDisplayName'])
    brand = product_name.split()[0] if product_name else 'Unknown Brand'
    
    # Construction de la description enrichie
    description_parts = [
        f"Product: {product_name}",
        f"Brand: {brand}",
        f"Type: {row['articleType']}",
        f"Category: {row['masterCategory']} - {row['subCategory']}",
        f"Color: {row['baseColour']}",
        f"Gender: {row['gender']}",
        f"Season: {row['season']}",
        f"Usage: {row['usage']}",
    ]
    
    # Ajouter l'année si disponible
    if row['year'] != 'Unknown' and pd.notna(row['year']):
        year_val = int(row['year']) if isinstance(row['year'], float) else row['year']
        description_parts.append(f"Year: {year_val}")
    
    return ". ".join(description_parts)


def create_langchain_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convertit le dataframe en liste de Documents LangChain avec métadonnées.
    
    Args:
        df: DataFrame nettoyé avec colonne 'description'
    
    Returns:
        Liste de Documents LangChain
    """
    documents = []
    
    for _, row in df.iterrows():
        # Métadonnées pour le filtrage et l'affichage
        metadata = {
            'id': int(row['id']),
            'product_name': str(row['productDisplayName']),
            'brand': str(row['productDisplayName']).split()[0] if row['productDisplayName'] else 'Unknown',
            'article_type': str(row['articleType']),
            'master_category': str(row['masterCategory']),
            'sub_category': str(row['subCategory']),
            'color': str(row['baseColour']),
            'gender': str(row['gender']),
            'season': str(row['season']),
            'usage': str(row['usage']),
            'year': str(row['year']),
            'image_url': str(row['link']) if 'link' in row and pd.notna(row['link']) else '',
            'filename': str(row['filename']) if 'filename' in row and pd.notna(row['filename']) else ''
        }
        
        doc = Document(
            page_content=row['description'],
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


# =============================================================================
# RAG RETRIEVAL FUNCTIONS
# =============================================================================

def format_rag_results(results: List[Tuple[Document, float]]) -> str:
    """
    Formate les résultats RAG pour inclusion dans le prompt.
    
    Args:
        results: Liste de tuples (Document, score)
    
    Returns:
        Contexte formaté pour le LLM
    """
    if not results:
        return "No relevant products found in the database."
    
    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        product_info = f"""
        Product {i}:
        - Name: {meta.get('product_name', 'N/A')}
        - Type: {meta.get('article_type', 'N/A')}
        - Color: {meta.get('color', 'N/A')}
        - Gender: {meta.get('gender', 'N/A')}
        - Season: {meta.get('season', 'N/A')}
        - Usage: {meta.get('usage', 'N/A')}
        - Category: {meta.get('master_category', 'N/A')} > {meta.get('sub_category', 'N/A')}
        - Image URL: {meta.get('image_url', 'Not available')}
        - Relevance Score: {score:.4f}
        """
        context_parts.append(product_info)
    
    return "\n".join(context_parts)


def extract_product_recommendations(results: List[Tuple[Document, float]]) -> List[Dict]:
    """
    Extrait les informations des produits recommandés pour l'affichage.
    
    Args:
        results: Liste de tuples (Document, score)
    
    Returns:
        Liste de dictionnaires avec les infos produits
    """
    recommendations = []
    
    for doc, score in results:
        meta = doc.metadata
        recommendations.append({
            'id': meta.get('id'),
            'name': meta.get('product_name', 'Unknown'),
            'brand': meta.get('brand', 'Unknown'),
            'type': meta.get('article_type', 'Unknown'),
            'color': meta.get('color', 'Unknown'),
            'gender': meta.get('gender', 'Unknown'),
            'season': meta.get('season', 'Unknown'),
            'usage': meta.get('usage', 'Unknown'),
            'category': f"{meta.get('master_category', '')} > {meta.get('sub_category', '')}",
            'image_url': meta.get('image_url', ''),
            'score': score
        })
    
    return recommendations


# =============================================================================
# GUARDRAILS - FASHION TOPIC DETECTION
# =============================================================================

# Keywords related to fashion and clothing
FASHION_KEYWORDS = {
    # English keywords
    'shirt', 'pants', 'jeans', 'dress', 'skirt', 'jacket', 'coat', 'sweater',
    'hoodie', 'blouse', 'suit', 'blazer', 'shorts', 'leggings', 'tights',
    'shoes', 'boots', 'sneakers', 'sandals', 'heels', 'flats', 'loafers',
    'hat', 'cap', 'beanie', 'scarf', 'gloves', 'belt', 'tie', 'bow tie',
    'bag', 'handbag', 'purse', 'backpack', 'wallet', 'watch', 'jewelry',
    'necklace', 'bracelet', 'earrings', 'ring', 'sunglasses', 'glasses',
    'underwear', 'bra', 'socks', 'swimwear', 'bikini', 'trunks',
    'casual', 'formal', 'sportswear', 'athletic', 'vintage', 'modern',
    'cotton', 'leather', 'denim', 'silk', 'wool', 'polyester', 'linen',
    'fashion', 'style', 'outfit', 'clothing', 'clothes', 'wear', 'apparel',
    'color', 'colour', 'size', 'fit', 'brand', 'designer', 'collection',
    'summer', 'winter', 'spring', 'fall', 'autumn', 'season', 'seasonal',
    'men', 'women', 'unisex', 'kids', 'children', 'boys', 'girls',
    'office', 'party', 'wedding', 'sports', 'gym', 'running', 'yoga',
    't-shirt', 'tshirt', 'polo', 'cardigan', 'vest', 'waistcoat',
    'trousers', 'chinos', 'joggers', 'tracksuit', 'romper', 'jumpsuit',
    
    # French keywords
    'chemise', 'pantalon', 'robe', 'jupe', 'veste', 'manteau', 'pull',
    'chaussures', 'bottes', 'baskets', 'sandales', 'talons', 'mocassins',
    'chapeau', 'casquette', 'bonnet', 'écharpe', 'gants', 'ceinture',
    'cravate', 'sac', 'sacoche', 'portefeuille', 'montre', 'bijoux',
    'collier', 'bracelet', 'boucles', 'bague', 'lunettes',
    'sous-vêtements', 'chaussettes', 'maillot', 'bain',
    'décontracté', 'élégant', 'sportif', 'vintage',
    'coton', 'cuir', 'soie', 'laine',
    'mode', 'style', 'tenue', 'vêtement', 'habit', 'habillement',
    'couleur', 'taille', 'marque', 'créateur',
    'été', 'hiver', 'printemps', 'automne', 'saison',
    'homme', 'femme', 'enfant', 'garçon', 'fille',
    'bureau', 'soirée', 'mariage', 'sport', 'gym', 'course',
    'polo', 'gilet', 'cardigan', 'survêtement', 'combinaison',
    'jean', 'short', 'legging', 'sweat', 'hoodie', 'blouson',
    'accessoire', 'accessoires', 'porter', 'acheter', 'cherche',
    'recommande', 'suggère', 'conseil', 'avis', 'article', 'produit'
}

# Patterns that indicate off-topic questions
OFF_TOPIC_PATTERNS = [
    r'\b(weather|météo|temps qu\'il fait)\b',
    r'\b(news|actualité|nouvelles)\b',
    r'\b(politics|politique|élection)\b',
    r'\b(recipe|recette|cuisine|cook)\b',
    r'\b(math|calcul|equation|équation)\b',
    r'\b(code|programming|programm|python|java|javascript)\b',
    r'\b(translate|traduire|traduction)\b',
    r'\b(history|histoire|historical)\b',
    r'\b(science|scientific|scientifique)\b',
    r'\b(medical|médical|health|santé|symptom|symptôme)\b',
    r'\b(travel|voyage|vacation|vacances)\b',
    r'\b(movie|film|series|série|music|musique)\b',
    r'\b(game|jeu|gaming)\b',
    r'\b(sport score|résultat|match)\b',
    r'\b(stock|bourse|investment|investissement)\b',
    r'\b(who is|qui est|what is|qu\'est-ce que)\b(?!.*?(wear|porter|fashion|mode|style|outfit|tenue))',
]


def is_fashion_related(query: str) -> Tuple[bool, float]:
    """
    Détermine si une requête est liée à la mode/vêtements.
    
    Args:
        query: La requête utilisateur
    
    Returns:
        Tuple (is_related: bool, confidence: float)
    """
    query_lower = query.lower()
    
    # Check for off-topic patterns first
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            # Exception: if fashion keywords are also present, it might be related
            fashion_count = sum(1 for kw in FASHION_KEYWORDS if kw in query_lower)
            if fashion_count < 2:
                return False, 0.9
    
    # Count fashion keywords
    fashion_count = sum(1 for kw in FASHION_KEYWORDS if kw in query_lower)
    
    # Calculate confidence based on keyword density
    words = query_lower.split()
    if len(words) == 0:
        return False, 0.0
    
    keyword_ratio = fashion_count / len(words)
    
    # Decision logic
    if fashion_count >= 2:
        return True, min(0.95, 0.6 + keyword_ratio)
    elif fashion_count == 1:
        return True, 0.5 + keyword_ratio * 0.3
    else:
        # No keywords found - might still be fashion-related
        # Allow for ambiguous queries to be processed
        return False, 0.3


def get_off_topic_response() -> str:
    """
    Retourne un message poli pour les questions hors sujet.
    
    Returns:
        Message de redirection
    """
    responses = [
        "🛍️ Je suis un assistant spécialisé dans la mode et les vêtements. "
        "Je peux vous aider à trouver des articles de mode comme des vêtements, "
        "chaussures, accessoires, etc. Comment puis-je vous aider avec votre style ?",
        
        "👗 Je suis dédié aux recommandations de mode ! "
        "Décrivez-moi le type de vêtement ou d'accessoire que vous recherchez, "
        "et je vous proposerai des articles adaptés.",
        
        "👔 Ma spécialité est la mode ! "
        "Dites-moi ce que vous cherchez (couleur, style, occasion, saison...) "
        "et je vous suggérerai des articles parfaits pour vous."
    ]
    import random
    return random.choice(responses)


def create_fashion_system_prompt() -> str:
    """
    Crée le prompt système pour le chatbot fashion RAG.
    
    Returns:
        Prompt système
    """
    return """You are a helpful fashion assistant specialized in recommending clothing and accessories. 
    Your role is to help users find the perfect fashion items based on their descriptions and preferences.

    IMPORTANT GUIDELINES:
    1. You ONLY answer questions related to fashion, clothing, shoes, and accessories.
    2. If a user asks about something unrelated to fashion, politely redirect them to fashion topics.
    3. When recommending products, use the context provided from the product database.
    4. Always mention the product name, color, type, and relevant details.
    5. Be enthusiastic and helpful about fashion recommendations.
    6. You can respond in the same language as the user (French or English).
    7. If no relevant products are found, suggest the user try different search terms.

    When presenting recommendations:
    - Highlight key features (color, style, usage)
    - Mention the brand if available
    - Suggest how the item could be styled or used
    - Be concise but informative

    Remember: You are a fashion expert assistant. Stay focused on fashion-related queries only."""


def create_rag_prompt_template() -> str:
    """
    Crée le template de prompt pour les requêtes RAG.
    
    Returns:
        Template de prompt
    """
    return """Based on the user's request and the following product recommendations from our database, 
    provide helpful fashion advice.

    PRODUCT DATABASE RESULTS:
    {context}

    USER REQUEST: {input}

    Please provide your fashion recommendation based on the products found. 
    If the products match the user's needs, describe them enthusiastically.
    If the products don't quite match, explain what's available and suggest alternatives.
    Always respond in the same language as the user's request.

    Your response:"""
