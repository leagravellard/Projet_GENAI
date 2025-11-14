import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# Charger les variables d'environnement (.env)
load_dotenv()

# =============== Modèle principal ===============

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
)

# =============== Définition des outils ===============

# 1. Recherche Web (DuckDuckGo)
search = DuckDuckGoSearchRun()

@tool
def recherche_web(query: str) -> str:
    """Recherche des informations récentes ou générales sur Internet."""
    return search.run(query)


# 2. Wikipedia
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=2000,
)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def recherche_wikipedia(query: str) -> str:
    """Recherche des informations factuelles sur Wikipedia."""
    return wiki.run(query)


# 3. Calculatrice simple en Python
@tool
def calculatrice(expression: str) -> str:
    """Évalue une expression mathématique (ex: '2 + 3 * 4')."""
    try:
        import math

        allowed_names = {
            k: getattr(math, k)
            for k in dir(math)
            if not k.startswith("_")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {e}"


tools = [recherche_web, recherche_wikipedia, calculatrice]


# =============== Fonction agent ===============

def appeler_agent(question: str) -> str:
    """
    Agent très simple :
    - 1er appel au LLM avec les outils déclarés (llm.bind_tools)
    - s'il demande un ou plusieurs tools, on les exécute
    - 2ème appel au LLM avec les résultats des tools pour obtenir la réponse finale
    """

    system_msg = SystemMessage(
        content=(
            "Tu es un assistant intelligent multi-outils. "
            "Tu peux utiliser : recherche_web, recherche_wikipedia, calculatrice. "
            "Utilise un outil uniquement si c'est vraiment utile. "
            "Donne des réponses claires et concises en français."
        )
    )
    human_msg = HumanMessage(content=question)

    # 1. Appel du LLM avec les outils
    llm_with_tools = llm.bind_tools(tools)
    ai_msg = llm_with_tools.invoke([system_msg, human_msg])

    # Si le modèle répond directement sans tool_call
    if not getattr(ai_msg, "tool_calls", None):
        return ai_msg.content

    # 2. Exécution des tools demandés
    tool_messages = []
    for tc in ai_msg.tool_calls:
        name = tc["name"]
        args = tc["args"] or {}

        tool_obj = next(t for t in tools if t.name == name)
        result = tool_obj.invoke(args)

        tool_messages.append(
            ToolMessage(
                content=str(result),
                name=name,
                tool_call_id=tc["id"],
            )
        )

    # 3. Deuxième appel au LLM avec les résultats des tools
    messages = [system_msg, human_msg, ai_msg] + tool_messages
    final_msg = llm.invoke(messages)
    return final_msg.content


# =============== Boucle interactive ===============

def test_agent():
    print("✅ Agent multi-outils prêt (web + Wikipedia + calculatrice).")
    print("Tape 'quitter' pour terminer.\n")

    while True:
        try:
            question = input("Votre question : ")
            if question.lower().strip() in {"q", "quit", "quitter", "exit"}:
                print("👋 Fin du test.")
                break

            reponse = appeler_agent(question)
            print("\n🧠 Réponse finale :", reponse, "\n")

        except KeyboardInterrupt:
            print("\n👋 Interruption manuelle. Fin du test.")
            break
        except Exception as e:
            print(f"❌ Une erreur est survenue : {e}\n")


if __name__ == "__main__":
    test_agent()
