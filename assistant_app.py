import streamlit as st
from dotenv import load_dotenv
# --- Imports LangChain ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- Charger les variables d'environnement ---
load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "chroma_db"

# --- Initialisation du modèle principal ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# --- Outil RAG : recherche dans la base Chroma ---
def search_documents(query: str) -> str:
    """Recherche dans les documents internes."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = db.as_retriever()
        
        # Création du prompt
        prompt = ChatPromptTemplate.from_template("""Réponds à la question suivante en te basant sur le contexte fourni :

Contexte : {context}

Question : {question}

Réponse :""")
        
        # Fonction pour formater les documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Création de la chaîne RAG avec LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)
    except Exception as e:
        return f"Erreur lors de la recherche dans les documents : {str(e)}"

# --- Autres outils ---
def search_web(query: str) -> str:
    """Recherche sur Internet."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"

def search_wikipedia(query: str) -> str:
    """Recherche sur Wikipedia."""
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        return wiki.run(query)
    except Exception as e:
        return f"Erreur lors de la recherche Wikipedia : {str(e)}"

def calculate_math(query: str) -> str:
    """Effectue des calculs mathématiques."""
    try:
        prompt = f"Résous ce problème mathématique et donne uniquement le résultat numérique : {query}"
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"

# --- Agent simplifié basé sur le LLM ---
def agent_query(user_input: str) -> str:
    """Agent qui décide quel outil utiliser et répond."""
    
    system_prompt = """Tu es un assistant intelligent avec accès à plusieurs outils :

1. search_documents : Pour rechercher dans les documents PDF internes
2. search_web : Pour rechercher des informations récentes sur Internet
3. search_wikipedia : Pour des informations encyclopédiques
4. calculate_math : Pour effectuer des calculs mathématiques

Pour chaque question :
- Analyse la question
- Décide quel outil utiliser (ou si tu peux répondre directement)
- Utilise l'outil si nécessaire
- Donne une réponse claire et précise

Si tu utilises un outil, commence ta réponse par [TOOL: nom_outil] suivi de la requête.
Exemples :
- "[TOOL: search_documents] Quels sont les chiffres de vente ?"
- "[TOOL: search_web] Dernières actualités IA"
- "[TOOL: calculate_math] 25 * 4 + 17"
- "[TOOL: search_wikipedia] Albert Einstein"

Si aucun outil n'est nécessaire, réponds directement."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    # Vérifier si l'agent veut utiliser un outil
    if "[TOOL:" in response_text:
        tool_start = response_text.find("[TOOL:") + 6
        tool_end = response_text.find("]", tool_start)
        tool_info = response_text[tool_start:tool_end].strip()
        
        # Extraire le nom de l'outil et la requête
        parts = tool_info.split("]", 1)
        if len(parts) == 2:
            tool_name = parts[0].strip()
            tool_query = parts[1].strip()
        else:
            tool_name = tool_info
            tool_query = user_input
        
        # Exécuter l'outil approprié
        if "search_documents" in tool_name:
            tool_result = search_documents(tool_query)
        elif "search_web" in tool_name:
            tool_result = search_web(tool_query)
        elif "search_wikipedia" in tool_name:
            tool_result = search_wikipedia(tool_query)
        elif "calculate_math" in tool_name:
            tool_result = calculate_math(tool_query)
        else:
            tool_result = "Outil non reconnu"
        
        # Demander au LLM de formuler la réponse finale
        final_messages = [
            SystemMessage(content="Tu es un assistant qui formule des réponses claires basées sur les résultats des outils."),
            HumanMessage(content=f"Question originale : {user_input}\n\nRésultat de l'outil : {tool_result}\n\nFormule une réponse claire et complète.")
        ]
        final_response = llm.invoke(final_messages)
        return final_response.content
    
    return response_text

# --- Interface Streamlit ---
st.set_page_config(page_title="Assistant Intelligent Multi-Compétences", page_icon="🤖")
st.title("🤖 Assistant Intelligent Multi-Compétences")
st.caption("Posez-moi des questions sur vos documents, le web, ou effectuez des calculs.")

# --- Historique de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if user_query := st.chat_input("Posez votre question ici..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            try:
                answer = agent_query(user_query)
            except Exception as e:
                answer = f"⚠️ Une erreur est survenue : {e}"
            st.markdown(answer)
    
    # Sauvegarde de la réponse
    st.session_state.messages.append({"role": "assistant", "content": answer})