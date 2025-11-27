# index_builder.py

"""
Script qui permet de charger les données prétraitée avec data_upload.py pour leur appliquer : 
1- chunking avec langchain_text_splitters
2- embedding avec Mistral
3- indexation avec faiss
4- enregistrement de l'index faiss et des metadonnées
"""

import json
import logging
import numpy as np
from pathlib import Path
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedder import Embedder
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IndexBuilder:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.embedder = Embedder()

        # Paths
        self.input_file = base_dir / "data" / "clean_data" / "data_events_clean.json"
        self.output_chunks_file = base_dir / "data" / "clean_data" / "data_events_chunks.json"
        self.output_vectors_file = base_dir / "data" / "clean_data" / "data_events_vectors.json"
        self.output_metadata_file = base_dir / "data" / "clean_data" / "metadata.json"
        self.emb_npy_file = base_dir / "data" / "clean_data" / "embeddings.npy"

        self.faiss_index_dir = base_dir / "data" / "clean_data" / "faiss_index"
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_file = self.faiss_index_dir / "faiss_index.bin"

    # -------------------------------------------------------------
    # 1. CHUNKING
    # -------------------------------------------------------------
    def chunk_event(self, events, chunk_size=800, chunk_overlap=100):
        """
        Découper les descriptions longues en chunks avec RecursiveCharacterTextSplitter.
        Crée un texte COMPLET combinant toutes les infos essentielles pour l'embedding.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )

        chunks = []

        for event in events:
            if not isinstance(event, dict):
                logger.warning("⚠️ Événement ignoré car il n'est pas un dictionnaire : %s", event)
                continue

            keywords = event.get("keywords_fr")
            if not isinstance(keywords, list):
                keywords = []

            description_complete = event.get("description_full_fr", "")

            if not description_complete or len(description_complete) <= chunk_size:
                chunk_texts = [description_complete]
            else:
                chunk_texts = text_splitter.split_text(description_complete)

            uid = event.get("uid", f"unknown")
            for idx, chunk_text in enumerate(chunk_texts):
                full_text = (
                    f"Titre : {event.get('title_fr','')}\n"
                    f"Ville : {event.get('location_city','')}\n"
                    f"Lieu : {event.get('location_name','')}\n"
                    f"Adresse : {event.get('location_address','')}\n"
                    f"Coordonnées : {event.get('location_coordinates')}\n"
                    f"Date de début : {event.get('firstdate_begin','')}\n"
                    f"Date de fin : {event.get('firstdate_end','')}\n"
                    f"Mots-clés : {', '.join(keywords)}\n\n"
                    f"Contenu : {chunk_text}"
                )

                chunks.append({
                    **event,
                    "chunk_id": f"{uid}_{idx}",
                    "chunk_index": idx,
                    "total_chunks": len(chunk_texts),
                    "description_chunk": chunk_text,
                    "full_text": full_text,
                })

        logger.info(f"📦 {len(chunks)} chunks créés à partir de {len(events)} événements")

        # Sauvegarde intermédiaire des chunks
        with open(self.output_chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Chunks sauvegardés : {self.output_chunks_file}")

        return chunks

    # -------------------------------------------------------------
    # 2. EMBEDDING
    # -------------------------------------------------------------
    def generate_embeddings(self, chunks):
        texts = [c["full_text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Normalisation L2 pour FAISS (cosine similarity)
        faiss.normalize_L2(embeddings)

        # Sauvegarde embeddings
        np.save(self.emb_npy_file, embeddings)
        logger.info(f"💾 Embeddings sauvegardées : {self.emb_npy_file}")

        return embeddings

    # -------------------------------------------------------------
    # 3. FAISS INDEX
    # -------------------------------------------------------------
    def build_faiss(self, embeddings):
        if embeddings.size == 0:
            raise ValueError("Aucun embedding n'a été fourni pour FAISS.")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(self.faiss_index_file))
        logger.info(f"💾 Index FAISS sauvegardé : {self.faiss_index_file}")

        return index

    # -------------------------------------------------------------
    # 4. SAVE METADATA
    # -------------------------------------------------------------
    def save_metadata(self, metadata):
        with open(self.output_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Métadonnées sauvegardées : {self.output_metadata_file}")

    # -------------------------------------------------------------
    # 5. PIPELINE COMPLET
    # -------------------------------------------------------------
    def run(self):
        logger.info("🚀 Démarrage indexation...")

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("results", [])
        if not events:
            logger.warning("⚠️ Aucun événement trouvé dans le fichier source.")
            return

        # Étapes
        chunks = self.chunk_event(events)
        embeddings = self.generate_embeddings(chunks)

        self.save_metadata(chunks)
        self.build_faiss(embeddings)

        logger.info("🎉 Indexation FAISS terminée !")


if __name__ == "__main__":
    builder = IndexBuilder(Path(__file__).parent.resolve())
    builder.run()

