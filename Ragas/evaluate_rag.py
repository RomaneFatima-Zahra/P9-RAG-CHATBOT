"""
Script d'évaluation RAGAS à partir du fichier DATA_RAG_TEST.jsonl
Le fichier doit contenir :
- question (string)
- answer (string)
- contexts (liste de dict ou liste de strings)
- ground_truth (string)
"""

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv
from config import EMBEDDING_MODEL, MODEL_NAME 
from pathlib import Path
import os
from typing import List, Dict, Optional
import json
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from ragas.metrics import (answer_correctness, faithfulness, context_precision, context_recall)

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

RAGAS_FILE = BASE_DIR / "DATA_RAG_TEST.jsonl"
METRIC_FILE = BASE_DIR / "ragas_results.jsonl"


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ⬇️ Configuration des modèles pour RAGAS
llm = ChatMistralAI(
    model=MODEL_NAME,
    mistral_api_key=MISTRAL_API_KEY
)

embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    mistral_api_key=MISTRAL_API_KEY
)

print("📥 Chargement du fichier RAGAS…")
df = pd.read_json(RAGAS_FILE, lines=True)

required_cols = {"question", "answer", "contexts", "ground_truth"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Colonnes manquantes dans {RAGAS_FILE} : {missing}")

# Transformation answer → response (format attendu par RAGAS)
df = df.rename(columns={"answer": "response"})

# Convertir les contexts dict → chaînes
def normalize_context(ctx):
    if isinstance(ctx, dict):
        return " | ".join(
            f"{k}: {v}" for k, v in ctx.items() if v not in [None, ""]
        )
    return str(ctx)

df["contexts"] = df["contexts"].apply(
    lambda lst: [normalize_context(c) for c in lst]
)

# On crée le dataset RAGAS
dataset = Dataset.from_pandas(df)

print("🚀 Lancement de l'évaluation RAGAS…\n")
result = evaluate(
    dataset,
    metrics=[
        answer_correctness,    # Similarité avec le ground truth
        faithfulness,          # La réponse suit-elle réellement le contexte ?
        context_precision,     # Qualité du retrieval (résultats utiles)
        context_recall         # Les bons documents ont-ils bien été récupérés ?
    ],
    llm=llm,              # AJOUT : spécifier le LLM Mistral
    embeddings=embeddings  #AJOUT : spécifier les embeddings Mistral
)

print("📊 Résultats des métriques RAGAS :\n")
print(result)
print("🎯 Évaluation terminée.")


# Afficher les résultats

print("📊 RÉSULTATS DE L'ÉVALUATION RAGAS")
print(f"\n🎯 Faithfulness:       {result.to_pandas()['faithfulness'].mean():.3f}")
print(f"\n🎯 answer_correctness:   {result.to_pandas()['answer_correctness'].mean():.3f}")
print(f"\n🎯 Context Precision:  {result.to_pandas()['context_precision'].mean():.3f}")
print(f"\n🎯 Context Recall:     {result.to_pandas()['context_recall'].mean():.3f}")


# Sauvegarder les résultats en JSON

result_dict = {
    "faithfulness": float(result.to_pandas()['faithfulness'].mean()),
    "answer_correctness": float(result.to_pandas()['answer_correctness'].mean()),
    "context_precision": float(result.to_pandas()['context_precision'].mean()),
    "context_recall": float(result.to_pandas()['context_recall'].mean()),
}

with open(METRIC_FILE, "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés dans : {METRIC_FILE}")
