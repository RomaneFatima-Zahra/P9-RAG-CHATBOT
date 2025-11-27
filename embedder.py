
# embedder.py
"""
embedder.py
Classe utilitaire pour générer des embeddings avec l'API Mistral
Utilisable par :
  - index_builder.py (pour indexer)
  - vector_store.py (pour rechercher)
"""

import os
import numpy as np
from typing import List, Union
from dotenv import load_dotenv
from mistralai import Mistral
from config import EMBEDDING_MODEL
import logging

# Load API key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("⚠️ MISTRAL_API_KEY manquante dans .env")

logger = logging.getLogger(__name__)


class Embedder:
    """
    Gestionnaire d'embeddings Mistral compatible FAISS.
    Peut embedder un texte unique ou une liste de textes.
    """

    def __init__(self, api_key: str = MISTRAL_API_KEY, model: str = EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)

    # -------------------------------------------------------------
    # Embedding sur 1 texte
    # -------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """Retourne l'embedding d'un seul texte → np.ndarray shape (dim,)"""
        resp = self.client.embeddings.create(
            model=self.model,
            inputs=[text]
        )
        embedding = np.array(resp.data[0].embedding, dtype="float32")
        return embedding

    # -------------------------------------------------------------
    # Embedding sur liste de textes
    # -------------------------------------------------------------
    def embed_texts(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """
        Embedding en batch pour accélérer l'indexation.
        Retour : np.ndarray (nb_samples, dim)
        """
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"🔠 Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    inputs=batch
                )
                for item in resp.data:
                    all_vectors.append(item.embedding)
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'embedding du batch : {e}")
                continue  # Skip et continue

        if not all_vectors:
            raise ValueError("⚠️ Aucun embedding généré")

        return np.array(all_vectors, dtype="float32")



