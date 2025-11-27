#chatbot_rag.py

"""
Ce script implémente un Chatbot RAG (Retrieval-Augmented Generation) spécialisé dans 
les recommandations d'événements culturels à Paris.

Fonctionnalités principales :
1. Recherche vectorielle (VectorStore + FAISS) :
   - Index FAISS pour retrouver rapidement les événements proches d'une requête.
   - Métadonnées associées aux vecteurs (titre, description, lieu, dates, mots-clés, etc.).
   - Embeddings générés via Embedder/Mistral pour la similarité cosinus.
2. Modèle de génération (Mistral via LangChain) :
   - Génère des réponses en langage naturel à partir du contexte récupéré par FAISS.
   - Respecte un prompt strict pour formater les événements de manière claire et lisible.
3. Pipeline RAG (Retrieval-Augmented Generation) :
   - Étape 1 : récupération des événements pertinents depuis VectorStore.
   - Étape 2 : formatage du contexte pour le LLM.
   - Étape 3 : génération de la réponse finale via Mistral.
   - Optionnel : stockage des interactions dans `ragas_logs.jsonl` pour l'évaluation RAGAS.
4. Test manuel interactif :
   - Permet de poser des questions directement dans la console.
   - Affiche la réponse générée par le chatbot et enregistre l'interaction.

Structure générale du code :
- ChatbotRAG : classe principale, encapsule VectorStore, LLM et pipeline RAG.
- _setup_prompt : construction du prompt système pour guider la génération.
- _format_context : transforme les événements FAISS en texte lisible pour le LLM.
- _setup_chain : pipeline LangChain pour combiner retrieval + prompt + LLM.
- chat : méthode publique pour interroger le chatbot.
- log_for_ragas : journalisation des interactions pour RAGAS.
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import os
from typing import List, Dict, Optional
import logging
import json
from dotenv import load_dotenv
from config import EMBEDDING_MODEL, MODEL_NAME 

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from mistralai import Mistral
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from vectore_store import VectorStore


# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FAISS_INDEX_FILE = BASE_DIR / "data" / "clean_data" / "faiss_index" / "faiss_index.bin"
METADATA_FILE = BASE_DIR / "data" / "clean_data" / "metadata.json"

# --------------------------------------------------------------------
# ENV 
# --------------------------------------------------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("⚠️ MISTRAL_API_KEY manquante dans .env")


# --------------------------------------------------------------------
# Class Chatbot RAG
# --------------------------------------------------------------------

class ChatbotRAG: #ChatbotRAG : classe principale, encapsule VectorStore, LLM et pipeline RAG.
    """
    Chatbot RAG basé sur le VectorStore + Mistral LLM.
    """

    def __init__(
        self,
        base_dir: Path,
        top_k: int = 1,
        temperature: float = 0.3,
        similarity_threshold: float = 0.35 ):

        # Recherche vectorielle
        logger.info("📦 Initialisation du VectorStore...")
        self.vector_store = VectorStore(
            base_dir=base_dir,
            top_k=top_k,
            similarity_threshold= similarity_threshold
        )

        # Initialiser le modèle Mistral via LangChain LLM
        logger.info("🤖 Initialisation du modèle Mistral...")

        self.llm = ChatMistralAI(
            model= MODEL_NAME,
            mistral_api_key= MISTRAL_API_KEY,
            temperature= temperature)

        self.top_k = top_k

         # Créer les prompts
        self._setup_prompt()
        
        # Créer la chaîne de traitement LangChain
        self._setup_chain()

        self.last_interaction = None  # Utilisé pour RAGAS (mais pas exposé)
        
        logger.info("🤖 Chatbot RAG initialisé et prêt.")

# -------------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------------
    
    def _setup_prompt(self): #construction du prompt système pour guider la génération.
        self.system_prompt= """

Tu es un assistant spécialisé dans les recommandations d'événements culturels à Paris.

Ton rôle est de :

Recommander des événements culturels pertinents basés sur la demande de l'utilisateur.
Fournir une réponse courte, naturelle et utile.
Adapter tes recommandations au contexte et aux besoins exprimés.
Te baser uniquement sur les événements futurs fournis dans le CONTEXTE.
Si aucun événement ne correspond, l'expliquer clairement et proposer une alternative réaliste.

Règles de présentation :
Répondre uniquement en texte.
Ne jamais utiliser de Markdown, HTML, JSON ou tout autre format de code dans le texte de la réponse.
Ne jamais formater la réponse comme un bloc de code.
Ne jamais afficher les caractères spéciaux comme **, *, #, -, `, , ou \n et \n\n dans la réponse.
Ne jamais mentionner les mots Markdown, HTML, JSON ou code.
La réponse doit être rédigée uniquement avec du texte normal.

Structure attendue :
Présenter chaque événement comme un paragraphe unique composé de phrases complètes.
Chaque événement doit inclure obligatoirement : Titre, Lieu, Ville, Adresse, Dates, Description et Mots-clés s'ils existent.
S'il y a plusieurs événements, les séparer uniquement par une ligne vide réelle (une ligne entièrement vide générée avec Entrée).
Ne jamais encoder les retours à la ligne.
Ne jamais utiliser de listes, de titres, d’emoji ou d’énumérations numérotées.

Rappel important :
Ne jamais faire référence à la similarité, aux calculs, à l'algorithme, au système ou à une logique technique.

CONTEXTE (résultats de recherche) : {context}        

        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{question}"),
            ]
        )

# -------------------------------------------------------------------------
# Formatage du contexte pour le LLM
# -------------------------------------------------------------------------
    def _format_context(self, events: List[Dict]) -> str: #transforme les événements FAISS en texte lisible pour le LLM.
        """Formate les événements récupérés en contexte lisible pour le LLM"""
        if not events:
            return "Aucun événement correspondant trouvé dans la base de données."

        context_parts = []
        
        for i, event in enumerate(events, 1):
            # Extraire les informations essentielles
            title = event.get("title_fr", "Sans titre")
            description = event.get("description_chunk", "")
            city = event.get("location_city", "")
            venue = event.get("location_name", "")
            address = event.get("location_address", "")
            date_begin = event.get("firstdate_begin", "")
            date_end = event.get("firstdate_end", "")
            keywords = event.get("keywords_fr", [])
            
            # Construire le texte de l'événement
            event_text = f"""--- Événement {i} ---
            Titre : {title}
            Lieu : {venue}
            Ville : {city}
            Adresse : {address}
            Date de début : {date_begin}
            Date de fin : {date_end}
            Mots-clés : {', '.join(keywords) if keywords else 'N/A'}
            Description : {description}
            """
            context_parts.append(event_text)
        
        return ".".join(context_parts)

# -------------------------------------------------------------------------
# Chaîne RAG
# -------------------------------------------------------------------------
    def _setup_chain(self): #pipeline LangChain pour combiner retrieval + prompt + LLM.
        """
        Pipeline LangChain :
        question ➜ recherche FAISS ➜ prompt ➜ Mistral ➜ réponse finale
        """
         # Définir les composants de la chaîne

         # Retrieval : retourne les événements RAW + Formatage : transforme en texte lisible
      
        def retrieve_and_format(inputs: Dict) -> str:
            """Récupère les événements pertinents via FAISS"""
            query = inputs["question"]
            events = self.vector_store.search(query)

            # ⬇️ On garde le RAW context pour RAGAS
            return {
                "formatted_context": self._format_context(events),
                "raw_context": events,
                "question": inputs["question"],
                "history": inputs.get("history", [])
        }

        retriever = RunnableLambda(retrieve_and_format)

        # Étape 2 : Construction du prompt
        build_prompt = RunnableLambda(
            lambda x: {
                "raw_context": x["raw_context"],
                "_prompt": self.prompt.invoke({
                "context": x["formatted_context"],
                "question": x["question"],
                "history": x["history"]
            })
        })

        # Étape 3 : Appel LLM
        call_llm = RunnableParallel({
            "answer": lambda x: self.llm.invoke(x["_prompt"]),
            "raw_context": lambda x: x["raw_context"]
            })
        
        # Chaîne complète RAG
        self.rag_chain = retriever | build_prompt | call_llm

        self.output_parser = StrOutputParser()
        logger.info("⛓️ Chaîne RAG configurée")

# -------------------------------------------------------------------------
# Entrée utilisateur → Réponse finale
# ------------------------------------------------------------------------
    
    def chat(self, user_query: str,history: List[Dict] = None ) -> str: #méthode publique pour interroger le chatbot
        """
        Point d'entrée principal pour interagir avec le chatbot.
        
        Args:
            user_query: Question ou demande de l'utilisateur
            
        Returns:
            Réponse générée par le chatbot
        """
        logger.info(f"💬 Question utilisateur : {user_query}")
        
        try:

            # 1) Appel de la chaîne (renvoie dict avant StrOutputParser)
            result = self.rag_chain.invoke({"question": user_query, "history": history or []})
            

            # 2) Extraire le contexte FAISS pour RAGAS
            raw_context = result["raw_context"]

            # 3) Réponse textuelle du modèle
            answer = self.output_parser.invoke(result["answer"])  # 🔥 convertit seulement la sortie LLM en texte

            # 4) Stockage interne pour évaluation ultérieure
            self.last_interaction = {
                "question": user_query,
                "answer": answer,
                "contexts": raw_context
                }
            logger.info("✅ Réponse générée avec succès")

            self.log_for_ragas()
            logger.info("📝 Interaction enregistrée pour RAGAS")

            # 5) Réponse renvoyée à l’API
            return answer
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de réponse : {e}")
            return f"Désolé, une erreur s'est produite : {str(e)}"

 # -------------------------------------------------------------------------
    # Logging optionnel pour RAGAS
# -------------------------------------------------------------------------
    def log_for_ragas(self, filepath="ragas_logs.jsonl"): #journalisation des interactions pour RAGAS.
        if not self.last_interaction:
            return
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.last_interaction, ensure_ascii=False) + "\n")
        logger.info("📝 Interaction enregistrée pour RAGAS.")


# -----------------------------------------------------------------------------
# Test manuel
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base = Path(__file__).parent.resolve()
    bot = ChatbotRAG(base)

    while True:
        q = input("\n❓ Pose-moi une question (ou 'quit') : ")
        if q.lower() in {"quit", "exit"}:
            break
        answer = bot.chat(q)
        bot.log_for_ragas()
        print("\n🤖 Réponse :\n")
        print(answer)