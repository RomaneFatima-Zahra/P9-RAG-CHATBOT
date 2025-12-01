# 🧠 P9-RAG-CHATBOT — Assistant intelligent de recommandation d’événements culturels à Paris

## 📌 Description du projet
P9-RAG-CHATBOT est un assistant conversationnel intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)**, spécialisé dans les recommandations d'événements culturels à Paris.  
Le chatbot s’appuie sur des données réelles issues de l’**API OpenAgenda**, génère des embeddings, effectue une recherche sémantique via **FAISS**, puis produit des réponses contextualisées à l’aide du **LLM Mistral Large**.

🎯 Objectif : permettre à un utilisateur de poser des questions en langage naturel (ex. : _« Que faire ce week-end à Paris ? »_) et d’obtenir des suggestions pertinentes d’événements culturels.

---

## 🚀 Fonctionnalités
- Ingestion des données OpenAgenda (≈ 100 événements)
- Nettoyage, enrichissement et vectorisation des textes
- Stockage et recherche vectorielle via **FAISS**
- Pipeline RAG complet : retrieval + prompt + génération Mistral Large
- API REST **FastAPI**
- Conteneurisation via **Docker / docker-compose**
- Tests automatisés + **GitHub Actions**
- Évaluation de la qualité du chatbot via **RAGAS**

---

## 🗂️ Organisation du dépôt
```
P9-RAG-CHATBOT/
│
├── api.py                         # API REST FastAPI
├── chatbot_rag.py                 # Pipeline RAG
├── config.py                      # Paramètres du projet
├── data_upload.py                 # Téléchargement / nettoyage des données
├── embedder.py                    # Génération des embeddings
├── index_builder.py               # Construction / sauvegarde de l'index FAISS
├── vectore_store.py               # Recherche vectorielle via FAISS
│
├── data/
│   ├── data_brut/                 # Données brutes issues de l’API
│   └── clean_data/                # Données nettoyées prêtes à l’indexation
│
├── Ragas/
│   ├── DATA_RAG_TEST.jsonl        # Jeu de test annoté
│   ├── evaluate_rag.py            # Script d’évaluation RAGAS
│   └── ragas_results.json         # Résultats de l’évaluation
│
├── tests/                         # Tests automatisés
│   ├── test_api.py
│   ├── test_indexbuilder.py
│   ├── test_rag.py
│   ├── test_upload.py
│   └── test_vectorestore.py
│
├── dockerfile                     # Build de l'image Docker
├── docker-compose.yml             # Lancement de l’API en conteneur
│
├── pyproject.toml / poetry.lock   # Dépendances du projet
└── .github/workflows/             # Pipeline CI GitHub Actions
```

---

## ⚙️ Installation

### 🔹 Méthode 1 — En local (sans Docker)
```bash
git clone https://github.com/RomaneFatima-Zahra/P9-RAG-CHATBOT
cd P9-RAG-CHATBOT
poetry install        # ou pip install -r requirements.txt
python data_upload.py
python index_builder.py
python vectore_store.py
python chatbot_rag.py
uvicorn api:app --reload
```

### 🔹 Méthode 2 — Avec Docker
```bash
docker-compose up --build
```

📍 API accessible sur : http://localhost:8000  
📍 Documentation Swagger : http://localhost:8000/docs

---

## 🔗 Endpoints API
| Méthode | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Vérifie que l’API fonctionne |
| POST | `/ask` | Question utilisateur → réponse du chatbot |
| POST | `/rebuild` | Recharge la base vectorielle |
| POST | `/rebuild/full` | Recrée complètement la base vectorielle |

---

## 🧠 Architecture

data_upload.py — ingestion et nettoyage des données OpenAgenda

embedder.py — génération des embeddings pour chaque chunk d’événement

index_builder.py — construction et sauvegarde de l’index FAISS + métadonnées

vectore_store.py — gestion du vector store et des recherches sémantiques

chatbot_rag.py — pipeline RAG : retrieval + prompt + génération avec Mistral

api.py — interface HTTP REST exposant les fonctionnalités du chatbot

Dockerfile / docker-compose.yml — configuration de déploiement conteneurisé

tests/ — tests unitaires/fonctionnels + CI

Ragas/ - Evaluation de performance

Data/ - Stockage de données.

---

## ⚙️ Choix technologiques

Les technologies utilisées ont été sélectionnées pour répondre aux exigences d’un système RAG performant, modulaire et facilement déployable :

| Composant | Choix technologique | Justification |
|----------|---------------------|---------------|
| Framework API | **FastAPI** | Rapidité, typage natif, documentation Swagger automatique |
| Modèle d’embedding | **Mistral Embed** | Haute qualité d’encodage pour le français, coût maîtrisé |
| Base vectorielle | **FAISS — IndexFlatIP** | Recherche vectorielle rapide, open-source, adaptée au scale-up |
| LLM | **Mistral Large (latest)** | Excellentes performances en français, très faible hallucination en mode RAG |
| Orchestration RAG | **LangChain** | Facilite la construction du pipeline retrieval → prompt → génération |
| Conteneurisation | **Docker + docker-compose** | Reproductibilité, facilité de déploiement, portabilité |
| Tests et CI/CD | **Pytest + GitHub Actions** | Vérification automatique du fonctionnement du système |
| Évaluation RAG | **RAGAS** | Métriques objectives pour mesurer la qualité des réponses générées |

🎯 Ces choix permettent d’obtenir un système :
- robuste contre les hallucinations
- performant malgré un volume de données limité
- facile à améliorer et à déployer dans une version future de production

---

## 📊 Évaluation des performances (RAGAS)
| Métrique | Score |
|---------|-------|
| Faithfulness | **0.90** |
| Answer Correctness | **0.31** |
| Context Precision | **0.43** |
| Context Recall | **0.62** |

🔍 Excellent contrôle des hallucinations. Le chatbot reste fidèle aux sources.  
⚠️ La précision du contexte récupéré est perfectible (retrieval à optimiser).

---

## 🔮 Améliorations et perspectives
- Ajustement du chunking et enrichissement des données vectorisées  
- Ajout d’un **reranker** (cross-encoder) après FAISS  
- Mise à jour automatique des données OpenAgenda  
- Interface conversationnelle web  
- Déploiement cloud + monitoring  
- Scalabilité et montée en charge

---

## 🤝 Contribution
Les contributions sont les bienvenues : correction de bugs, optimisation du retrieval, nouvelles données, UI web…  
N’hésitez pas à ouvrir une issue ou une pull request.

---

## ✨ Auteur
Projet réalisé dans le cadre du **Projet 9 — OpenClassrooms**  
👤 *BARHOU Fatima-Zahra*
