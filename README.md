# Assistant Intelligent Multi-Compétences (RAG + Agents)

Ce projet est une application web conversationnelle construite avec Streamlit et LangChain. L'assistant est capable de répondre à des questions en utilisant plusieurs sources d'information : un corpus de documents locaux (PDF), une recherche web, Wikipedia ainsi qu'une calculatrice.

## 🌟 Fonctionnalités

- **Interface de Chat Interactive** : Une application web simple et intuitive construite avec Streamlit.
- **RAG sur Documents Locaux** : L'assistant peut lire et répondre à des questions sur le contenu des fichiers PDF que vous placez dans le dossier `documents`.
- **Agents Multi-Outils** : L'assistant utilise un agent intelligent (ReAct) pour choisir le meilleur outil afin de répondre à une question :
    - **Recherche Documents Internes** : L'outil RAG pour les questions spécifiques aux documents.
    - **Recherche Web** : Utilise DuckDuckGo pour les informations générales ou récentes.
    - **Wikipedia** : Pour les requêtes factuelles sur des sujets encyclopédiques.
    - **Calculatrice** : Pour effectuer des opérations mathématiques.
- **Mémoire Conversationnelle** : L'assistant se souvient des échanges précédents pour maintenir le contexte de la conversation.

## 🏗️ Architecture

L'application est centrée autour d'un **AgentExecutor** de LangChain.
1.  L'utilisateur envoie une question via l'interface Streamlit.
2.  L'AgentExecutor reçoit la question et l'historique de la conversation.
3.  Le LLM (GPT-4o) au cœur de l'agent analyse la requête et décide s'il a besoin d'un outil.
4.  Si un outil est nécessaire, l'agent l'appelle (par exemple, l'outil RAG pour une question sur un document).
5.  L'agent reçoit le résultat de l'outil et l'utilise pour formuler la réponse finale.
6.  S'il n'a pas besoin d'outil, le LLM répond directement.
7.  La réponse est affichée à l'utilisateur et la conversation est mémorisée.

## 🛠️ Stack Technique

- **Langage** : Python 3.11+
- **Frameworks Principaux** : LangChain, Streamlit
- **LLM** : OpenAI GPT-4o
- **Base de Données Vectorielle** : ChromaDB (pour le RAG)
- **Outils** : DuckDuckGo Search, Wikipedia, LLMMath

## 🚀 Installation et Lancement

Suivez ces étapes pour lancer le projet sur votre machine.

### 1. Prérequis

- Python 3.11 ou supérieur.
- Un compte OpenAI et une clé API.

### 2. Installation

1.  **Clonez le dépôt** (ou téléchargez les fichiers dans un dossier) :
    ```bash
    # git clone https://votre-url-de-depot.git
    # cd nom-du-dossier
    ```

2.  **Créez et activez un environnement virtuel** :
    ```bash
    # Créer l'environnement
    python -m venv venv

    # Activer sur Windows
    .\venv\Scripts\activate

    # Activer sur macOS/Linux
    # source venv/bin/activate
    ```

3.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurez votre clé API** :
    - Créez un fichier nommé `.env` à la racine du projet.
    - Ajoutez votre clé API OpenAI dans ce fichier :
      ```
      OPENAI_API_KEY="sk-VotreCleSecrete..."
      ```

### 3. Utilisation

1.  **Ajoutez vos documents** :
    - Placez tous les fichiers PDF que vous souhaitez analyser dans le dossier `documents`.

2.  **Ingérez les documents** :
    - Lancez ce script une seule fois (ou à chaque fois que vous modifiez les documents). Il va lire les PDF et les stocker dans la base de données vectorielle.
    ```bash
    python ingest.py
    ```

3.  **Lancez l'application** :
    - Une fois l'ingestion terminée, lancez l'application Streamlit.
    ```bash
    streamlit run app.py
    ```
    ou 
    ```bash
    streamlit run assistant_app.py
    ```
    - Votre navigateur devrait s'ouvrir sur l'interface de l'assistant.

## 📁 Structure du Projet

```
.
├── 📄 app.py               # Script principal de l'application Streamlit
├── 📄 assistant_app.py     # Script alternatif de l'application Streamlit
├── 📄 ingest.py            # Script pour l'ingestion des documents
├── 📄 purger.py            # Script pour purger Chroma_db
├── 📄 test_agent.py        # Script de test pour l'agent en ligne de commande
├── 📄 requirements.txt     # Liste des dépendances Python
├── 📄 .env                 # Fichier pour les variables d'environnement (clé API)
├── 📄 .gitignore           # Fichiers et dossiers à ignorer par Git
├── 📁 documents/           # Dossier où placer vos fichiers PDF
└── 📁 chroma_db/           # Base de données vectorielle (créée par ingest.py)
```
