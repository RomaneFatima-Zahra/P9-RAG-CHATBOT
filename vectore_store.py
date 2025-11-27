# vector_store.py
"""
VectorStore
Système de recherche vectorielle pour le chatbot RAG.

Charge :
  - Index FAISS
  - Fichier metadata.json
  - Embedder (embeddings des requêtes)

Permet :
  - Recherche d'événements culturels par similarité cosinus
  - Renvoie les résultats sous forme de dictionnaires exploitables par le chatbot
"""

import json
import logging
import numpy as np
from pathlib import Path
import faiss
from typing import List, Dict, Optional
from embedder import Embedder
import os
from dotenv import load_dotenv
from mistralai import Mistral
from config import EMBEDDING_MODEL


# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Classe VectorStoreRAG
# --------------------------------------------------------------------

class VectorStore:
    """
    Classe de stockage vectoriel utilisant FAISS pour la recherche d'événements culturels.

    - charge l'index FAISS
    - charge les métadonnées alignées
    - génère les embeddings des requêtes (Mistral) avec Embedder
    - renvoie les événements les plus proches
    """

    def __init__(
        self,
        base_dir: Path,
        faiss_file: str = "data/clean_data/faiss_index/faiss_index.bin",
        metadata_file: str = "data/clean_data/metadata.json",
        similarity_threshold: float = 0.35,
        top_k: int = 3,
    ):
        self.base_dir = base_dir
        self.faiss_index_path = base_dir / faiss_file
        self.metadata_path = base_dir / metadata_file
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Charger FAISS
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"❌ Index FAISS introuvable : {self.faiss_index_path}")
        self.index = faiss.read_index(str(self.faiss_index_path))
        logger.info(f"📌 Index FAISS chargé ({self.index.ntotal} vecteurs)")

        # Charger metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"❌ Metadata introuvable : {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)
        logger.info(f"📌 Métadonnées chargées ({len(self.metadata)} items)")

        if len(self.metadata) != self.index.ntotal:
            logger.warning(
                f"⚠️ Désalignement possible : {len(self.metadata)} metadata ≠ {self.index.ntotal} vecteurs FAISS"
            )

        # Embedding provider
        self.embedder = Embedder()

        logger.info("✅ VectorStore prêt")

    # -------------------------------------------------------------------------
    # Recherche vectorielle
    # -------------------------------------------------------------------------
    def search(self, query: str) -> List[Dict]:
        """
        Recherche par similarité cosinus dans la base FAISS.
        Retourne une liste de dictionnaires correspondant aux chunks d'événements.
        """
        logger.info(f"🔍 Recherche d'événements pour : '{query}'")

        # 1) Embedding de la requête
        q_vec = self.embedder.embed_text(query).astype("float32")

        # 2) Normalisation required pour cosine similarity
        q_vec = np.expand_dims(q_vec, axis=0)
        faiss.normalize_L2(q_vec)

        # 3) Recherche FAISS
        distances, indices = self.index.search(q_vec, self.top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score < self.similarity_threshold:
                continue
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx].copy()
            item["similarity_score"] = float(score)
            results.append(item)

        logger.info(f"📊 {len(results)} résultats renvoyés")
        return results
    
    # -------------------------------------------------------------------------
    # Fonction pour le rebuild dans l'API
    # -------------------------------------------------------------------------
    
    def rebuild(self):
        """
        Recharge l'index FAISS et les métadonnées depuis le disque.
        """
        self.index = faiss.read_index(str(self.faiss_index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)


# -----------------------------------------------------------------------------
# Optionnel : outil d'affichage debug (console)
# -----------------------------------------------------------------------------
def display_results(query: str, results: List[Dict], max_desc_length: int = 300):
    """
    Affichage console des résultats (debug).
    N'est PAS utilisé par le chatbot, mais utile pour valider FAISS.
    """
    print("\n" + "=" * 80)
    print(f"🔍 REQUÊTE : {query}")
    print("=" * 80)

    if not results:
        print("⚠️ Aucun résultat trouvé")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\n{'─' * 80}")
        print(f"🏆 Rang #{rank} | Score: {r['similarity_score']:.4f}")
        print(f"📌 Titre : {r.get('title_fr', '—')}")
        print(f"📍 Lieu : {r.get('location_name', '—')}, {r.get('location_city', '—')}")
        print(f"📅 Du {r.get('firstdate_begin', '—')} au {r.get('firstdate_end', '—')}")

        desc = r.get("description_chunk", "")
        if desc:
            preview = desc[:max_desc_length] + ("..." if len(desc) > max_desc_length else "")
            print(f"📄 Description : {preview}")

        keywords = r.get("keywords_fr")
        if isinstance(keywords, list) and keywords:
            print(f"🏷️ Mots-clés : {', '.join(keywords[:5])}")


# -----------------------------------------------------------------------------
# Test manuel (si exécuté directement)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base = Path(__file__).parent.resolve()
    store = VectorStore(base)

    q_test = [
        "festival",
        "spectacle de théâtre",
        "spectacle pour enfants",
        "escape game"]
    

    for q in q_test:
        results = store.search(q)
        display_results(q, results)

    results = store.search(q)
    display_results(q, results)

