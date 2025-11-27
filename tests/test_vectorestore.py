"""
Script de test simple pour VectorStore
Teste la recherche avec différentes requêtes et affiche les résultats
"""
import json
from pathlib import Path
from vectore_store import VectorStore , display_results

def main():
    print("\n=================== TEST VECTOR STORE ===================\n")

    # 📁 base_dir = dossier du projet (où se trouve data/clean_data/)
    base_dir = Path(__file__).resolve().parents[1]   # remonte à la racine du projet
    print(f"📌 Base dir utilisé pour VectorStore : {base_dir}")

    store = VectorStore(base_dir)

    queries = [
        "festival",
        "spectacle de théâtre",
        "spectacle pour enfants",
        "escape game" ]

    for q in queries:
        print("\n-------------------------------------------------------------")
        results = store.search(q)
        display_results(q, results)

    print("\n🎉 TEST VECTOR STORE TERMINÉ !\n")

if __name__ == "__main__":
    main()