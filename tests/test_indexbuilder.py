# test_indexbuilder.py

"""
Test complet :
1. Récupération du fichier test_clean_3_events.json du test_upload.py
2. Chunking via IndexBuilder.chunk_event
3. Génération d'embeddings
4. Construction de l'index FAISS
"""

import json
from pathlib import Path
from index_builder import IndexBuilder


def load_test_events():
    """
    Charge le fichier généré par test_upload.py :
    test_clean_3_events.json
    """
    file = Path("data/clean_data/test_clean_3_events.json")

    if not file.exists():
        raise FileNotFoundError(
            f"❌ Le fichier {file} est introuvable. "
            f"Tu dois d'abord exécuter test_upload.py !"
        )

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("results", [])
    if len(events) != 3:
        raise ValueError(f"❌ Le fichier ne contient pas 3 événements mais {len(events)}")

    print(f"📥 3 événements chargés depuis : {file}")
    return events


def test_chunk_and_embeddings():
    print("\n=== TEST : Chunking + Embeddings + FAISS ===")

    base_dir = Path.cwd()  # dossier actuel
    Builder = IndexBuilder(base_dir)

    # 1. Charger les événements
    events = load_test_events()

    # 2. Chunking
    chunks = Builder.chunk_event(events) #chunking avec la fonction issue de IndexBuilder

    print(f"📦 {len(chunks)} chunks générés") #résultat : 7 chunks créés à partir de 3 événements

    assert len(chunks) >= 3, "❌ Le chunking doit générer au moins un chunk par événement"

    # 3. Embeddings
    embeddings = Builder.generate_embeddings(chunks) #générer les embedding avec la fonction issue de IndexBuilder
    print(f"🔢 Embeddings shape : {embeddings.shape}") #résultat :  Embeddings shape : (7, 1024)

    assert embeddings.shape[0] == len(chunks), "❌ Nombre d'embeddings ≠ nombre de chunks"

    # 4. FAISS
    index = Builder.build_faiss(embeddings) #Indexation avec la fonction issue de IndexBuilder
    print("🎯 Index FAISS construit avec succès")

    assert index.ntotal == embeddings.shape[0], "❌ FAISS n'a pas indexé tous les embeddings"

    print("\n🎉 TEST GLOBAL RÉUSSI — Chunking + Embeddings + FAISS\n")

    # 5. Save metadata 

    metadata_test = Builder.save_metadata


if __name__ == "__main__":
    test_chunk_and_embeddings()
