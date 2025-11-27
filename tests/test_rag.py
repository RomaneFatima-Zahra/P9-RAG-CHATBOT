#test_rag.py
"""
Script de test du sytème RAG avec la génération des réponses avec le Chatbot ! 

"""

import json
from pathlib import Path
from chatbot_rag import ChatbotRAG


def test_chatbot_rag():
    print("\n=================== TEST CHATBOT RAG ===================\n")
    
    # 📁 base_dir = dossier du projet (où se trouve data/clean_data/)
    base_dir = Path(__file__).resolve().parents[1]
    print(f"📌 Base dir utilisé : {base_dir}")

    # Initialisation du Chatbot
    bot = ChatbotRAG(base_dir=base_dir, top_k=2, similarity_threshold=0.3)

    # Requêtes de test
    test_queries = [
        "festival",
        "spectacle de théâtre",
        "spectacle pour enfants",
        "escape game"]
    

    for q in test_queries:
        print("\n-------------------------------------------------------------")
        print(f"❓ Question : {q}")
        answer = bot.chat(q)
        print(f"\n🤖 Réponse :\n{answer}")

    print("\n🎉 TEST CHATBOT RAG TERMINÉ !\n")

if __name__ == "__main__":
    test_chatbot_rag()
