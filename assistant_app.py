import streamlit as st
from dotenv import load_dotenv
import re
# --- Imports LangChain ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
import wikipedia

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
        
        # Vérifier si la base contient des documents
        collection = db.get()
        if not collection['ids']:
            return "⚠️ La base de données est vide. Veuillez d'abord indexer des documents."
        
        print(f"[DEBUG RAG] Nombre de documents dans la base : {len(collection['ids'])}")
        
        # Recherche de similarité directe
        docs = db.similarity_search(query, k=3)
        print(f"[DEBUG RAG] Documents trouvés : {len(docs)}")
        
        if not docs:
            return "⚠️ Aucun document pertinent trouvé dans la base."
        
        # Formater le contexte
        context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        print(f"[DEBUG RAG] Longueur du contexte : {len(context)} caractères")
        
        # Création du prompt
        prompt = ChatPromptTemplate.from_template("""Réponds à la question suivante en te basant UNIQUEMENT sur le contexte fourni.
Si le contexte ne contient pas l'information, dis-le clairement.

Contexte : {context}

Question : {question}

Réponse :""")
        
        # Création de la chaîne simplifiée
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"context": context, "question": query})
        print(f"[DEBUG RAG] Réponse générée : {result[:100]}...")
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DEBUG RAG] Erreur complète :\n{error_details}")
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
        # Configurer la langue française
        wikipedia.set_lang("fr")
        
        # Rechercher la page
        try:
            # Essayer d'abord une recherche exacte
            page = wikipedia.page(query, auto_suggest=True)
        except wikipedia.exceptions.DisambiguationError as e:
            # Si plusieurs résultats, prendre le premier
            page = wikipedia.page(e.options[0])
        except wikipedia.exceptions.PageError:
            # Si la page n'existe pas, chercher des suggestions
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return f"Aucun résultat trouvé sur Wikipedia pour : {query}"
            page = wikipedia.page(search_results[0])
        
        # Limiter le contenu à 2000 caractères
        summary = page.summary[:2000]
        if len(page.summary) > 2000:
            summary += "..."
        
        return f"**{page.title}**\n\n{summary}\n\n🔗 URL : {page.url}"
        
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
- Utilise TOUJOURS search_documents en priorité si la question concerne des informations qui pourraient être dans des documents internes

IMPORTANT : Pour utiliser un outil, tu DOIS répondre EXACTEMENT dans ce format :
TOOL: nom_outil
QUERY: ta requête ici

Exemples :
Question sur des documents internes → 
TOOL: search_documents
QUERY: Quels sont les chiffres de vente ?

Question d'actualité → 
TOOL: search_web
QUERY: Dernières actualités IA

Question encyclopédique → 
TOOL: search_wikipedia
QUERY: Albert Einstein

Calcul mathématique → 
TOOL: calculate_math
QUERY: 25 * 4 + 17

Si aucun outil n'est nécessaire, réponds directement SANS utiliser le format TOOL/QUERY."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    # Détecter si l'agent veut utiliser un outil avec le nouveau format
    tool_pattern = r'TOOL:\s*(\w+)\s*\nQUERY:\s*(.+?)(?:\n|$)'
    match = re.search(tool_pattern, response_text, re.DOTALL)
    
    if match:
        tool_name = match.group(1).strip()
        tool_query = match.group(2).strip()
        
        # Exécuter l'outil approprié
        st.info(f"🔧 Utilisation de l'outil : **{tool_name}**\n\nRequête : *{tool_query}*")
        
        if tool_name == "search_documents":
            tool_result = search_documents(tool_query)
        elif tool_name == "search_web":
            tool_result = search_web(tool_query)
        elif tool_name == "search_wikipedia":
            tool_result = search_wikipedia(tool_query)
        elif tool_name == "calculate_math":
            tool_result = calculate_math(tool_query)
        else:
            tool_result = f"⚠️ Outil '{tool_name}' non reconnu"
        
        # Demander au LLM de formuler la réponse finale
        final_messages = [
            SystemMessage(content="Tu es un assistant qui formule des réponses claires basées sur les résultats des outils. Réponds en français."),
            HumanMessage(content=f"Question originale : {user_input}\n\nRésultat de l'outil : {tool_result}\n\nFormule une réponse claire et complète en français.")
        ]
        final_response = llm.invoke(final_messages)
        return final_response.content
    
    return response_text

# --- Interface Streamlit ---
st.set_page_config(page_title="Assistant Intelligent Multi-Compétences", page_icon="🤖")
st.title("🤖 Assistant Intelligent Multi-Compétences")
st.caption("Posez-moi des questions sur vos documents, le web, ou effectuez des calculs.")

# Sidebar pour les informations de debug
with st.sidebar:
    st.header("🔍 Informations de Debug")
    if st.button("Vérifier la base RAG"):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            collection = db.get()
            st.success(f"✅ Base RAG chargée : {len(collection['ids'])} documents")
            if collection['ids']:
                st.write("**Premiers IDs:**", collection['ids'][:5])
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

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