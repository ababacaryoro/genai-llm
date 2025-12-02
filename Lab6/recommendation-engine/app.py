"""
Fashion RAG Chatbot Application
A Streamlit chatbot with RAG capabilities for fashion product recommendations
Compatible with LangChain v0.2+ and ChromaDB
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
# LangChain Core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Local utilities
from utils import (
    count_tokens,
    #load_vectorstore, 
    estimate_cost,
    is_fashion_related,
    get_off_topic_response,
    create_fashion_system_prompt,
    format_rag_results,
    extract_product_recommendations
)


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ChromaDB Configuration
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fashion_products"

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fashion RAG Chatbot",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .main-header { text-align: center; padding: 1rem; }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .product-image {
        max-width: 100%;
        height: 200px;
        object-fit: contain;
        border-radius: 8px;
    }
    .metadata-tag {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 12px;
        margin: 2px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COST TRACKING
# =============================================================================

def initialize_cost_tracker():
    return {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0
    }


def update_costs(costs: dict, input_tokens: int, output_tokens: int, model: str):
    estimated_cost = estimate_cost(input_tokens, output_tokens, model)
    
    costs["input_tokens"] += input_tokens
    costs["output_tokens"] += output_tokens
    costs["total_tokens"] += (input_tokens + output_tokens)
    costs["total_cost_usd"] += estimated_cost
    
    return costs


# =============================================================================
# RAG INITIALIZATION
# =============================================================================

documents = None
documents = []
with open("data/docs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        documents.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata", {})))

MAX_DOCS = 5000  # A Ajuster selon les besoins et le quota API
docs_to_index = documents[:MAX_DOCS]

@st.cache_resource
def load_vectorstore():
    """
    Charge la base vectorielle ChromaDB.
    Utilise le cache Streamlit pour éviter les rechargements.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("embeddings initialized")  # Debug log
        vectorstore = Chroma.from_documents(
            documents=docs_to_index,
            embedding=embeddings,
            #persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        
        
        # Vérifier que la collection n'est pas vide
        collection_count = vectorstore._collection.count()
        if collection_count == 0:
            return None, "La base vectorielle est vide. Exécutez d'abord le notebook de préparation."
        
        return vectorstore, f"✅ {collection_count} produits chargés"
    
    except Exception as e:
        return None, f"❌ Erreur de chargement: {str(e)}"


def retrieve_products(vectorstore, query: str, k: int = 5):
    """
    Recherche les produits similaires dans la base vectorielle.
    
    Args:
        vectorstore: Instance ChromaDB
        query: Requête utilisateur
        k: Nombre de résultats
    
    Returns:
        Liste de tuples (Document, score)
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        st.error(f"Erreur de recherche: {str(e)}")
        return []


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def initialize_llm(model_name: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def initialize_memory() -> BaseChatMessageHistory:
    return ChatMessageHistory()


def build_rag_chain(llm: ChatOpenAI):
    """
    Construit la chaîne RAG avec le prompt fashion.
    """
    system_prompt = create_fashion_system_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", """Context from product database:
{context}

User request: {input}

Please provide helpful fashion recommendations based on the products found.""")
    ])
    
    return prompt | llm


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_product_cards(recommendations: list):
    """
    Affiche les cartes de produits recommandés avec images.
    """
    if not recommendations:
        return
    
    cols = st.columns(min(len(recommendations), 3))
    
    for i, product in enumerate(recommendations[:6]):  # Max 6 produits
        col_idx = i % 3
        with cols[col_idx]:
            with st.container():
                st.markdown(f"### {product['name'][:50]}...")
                
                # Afficher l'image si disponible
                if product['image_url']:
                    try:
                        st.image(
                            product['image_url'], 
                            caption=f"{product['type']} - {product['color']}"
                        )
                    except Exception as e:
                        st.info("🖼️ Image non disponible")
                        print("Image load error:", str(e))
                else:
                    st.info("🖼️ Image non disponible")
                
                # Métadonnées dans un expander
                with st.expander("📋 Détails du produit"):
                    st.markdown(f"""
                    - **Marque:** {product['brand']}
                    - **Type:** {product['type']}
                    - **Couleur:** {product['color']}
                    - **Genre:** {product['gender']}
                    - **Saison:** {product['season']}
                    - **Usage:** {product['usage']}
                    - **Catégorie:** {product['category']}
                    - **Score de pertinence:** {product['score']:.4f}
                    """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.markdown("<h1 class='main-header'>👗 Fashion RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Votre assistant mode intelligent - Décrivez ce que vous cherchez!</p>", unsafe_allow_html=True)
    
    # ------------------------------------
    # SIDEBAR
    # ------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_name = st.selectbox(
            "Modèle LLM",
            options=["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"],
            index=0,
        )
        
        temperature = st.slider("Température", 0.0, 1.0, 0.7, 0.1)
        
        num_results = st.slider(
            "Nombre de suggestions",
            min_value=1,
            max_value=10,
            value=5,
            help="Nombre de produits à rechercher dans la base"
        )
        
        st.divider()
        
        # Charger la base vectorielle
        st.header("📦 Base de données")
        vectorstore, status_msg = load_vectorstore()
        
        if vectorstore:
            st.success(status_msg)
        else:
            st.error(status_msg)
            st.warning("⚠️ Exécutez le notebook `rag_preparation.ipynb` pour créer la base vectorielle.")
        
        st.divider()
        
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = initialize_memory()
            st.session_state.costs = initialize_cost_tracker()
            st.session_state.last_recommendations = []
            st.rerun()
        
        st.divider()
        
        st.header("📊 Statistiques")
        if "costs" in st.session_state:
            st.metric("Total Tokens", st.session_state.costs["total_tokens"])
            st.metric("Input Tokens", st.session_state.costs["input_tokens"])
            st.metric("Output Tokens", st.session_state.costs["output_tokens"])
            st.metric("Coût estimé (USD)", f"${st.session_state.costs['total_cost_usd']:.4f}")
        
        st.divider()
        st.caption("💡 Exemples de requêtes:")
        st.caption("- 'Je cherche une robe rouge pour l'été'")
        st.caption("- 'Blue casual shoes for men'")
        st.caption("- 'Veste noire élégante pour le bureau'")
    
    # -------------------------
    # SESSION STATE
    # -------------------------
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()
    
    if "costs" not in st.session_state:
        st.session_state.costs = initialize_cost_tracker()
    
    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = []
    
    # Init model + chain
    llm = initialize_llm(model_name, temperature)
    chain = build_rag_chain(llm)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Afficher les produits si c'est une réponse assistant avec des recommandations
            if msg["role"] == "assistant" and "recommendations" in msg:
                display_product_cards(msg["recommendations"])
    
    # -------------------------
    # CHAT INPUT
    # -------------------------
    
    if user_input := st.chat_input("Décrivez ce que vous cherchez... (ex: 'robe bleue pour l'été')"):
        
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.memory.add_user_message(user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            # Vérifier si la requête est liée à la mode
            is_fashion, confidence = is_fashion_related(user_input)
            
            if not is_fashion and confidence > 0.7:
                # Requête hors sujet - afficher message de redirection
                off_topic_response = get_off_topic_response()
                st.markdown(off_topic_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": off_topic_response
                })
                st.session_state.memory.add_ai_message(off_topic_response)
            
            else:
                # Requête liée à la mode - utiliser le RAG
                with st.spinner("🔍 Recherche des articles correspondants..."):
                    try:
                        # Vérifier si la base vectorielle est chargée
                        if vectorstore is None:
                            st.error("❌ La base de données n'est pas disponible. Veuillez d'abord exécuter le notebook de préparation.")
                            st.stop()
                        
                        # Recherche RAG
                        rag_results = retrieve_products(vectorstore, user_input, k=num_results)
                        
                        if not rag_results:
                            context = "No products found matching the query."
                            recommendations = []
                        else:
                            context = format_rag_results(rag_results)
                            recommendations = extract_product_recommendations(rag_results)
                        
                        # Invoquer le LLM avec le contexte RAG
                        response = chain.invoke({
                            "input": user_input,
                            "context": context,
                            "history": st.session_state.memory.messages,
                        })
                        
                        ai_text = response.content
                        
                        # Update costs
                        usage = response.usage_metadata or {}
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        
                        st.session_state.costs = update_costs(
                            st.session_state.costs,
                            input_tokens,
                            output_tokens,
                            model_name
                        )
                        
                        # Afficher la réponse
                        st.markdown(ai_text)
                        
                        # Afficher les cartes produits
                        if recommendations:
                            st.markdown("---")
                            st.markdown("### 🛍️ Articles recommandés:")
                            display_product_cards(recommendations)
                        
                        # Sauvegarder
                        st.session_state.memory.add_ai_message(ai_text)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ai_text,
                            "recommendations": recommendations
                        })
                        st.session_state.last_recommendations = recommendations
                    
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
