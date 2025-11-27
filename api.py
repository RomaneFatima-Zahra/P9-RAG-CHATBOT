from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from chatbot_rag import ChatbotRAG, BASE_DIR
from index_builder import IndexBuilder
import uvicorn

rag_bot = ChatbotRAG(base_dir=BASE_DIR)

app = FastAPI(
    title=" Je suis votre assistant intelligent pour vous recommander des événements culturels autour de Paris ",
    description="API Fastapi qui permet d'utiliser le chatbot intelligent qui est capable de fournir des recommandations personnalisées basées sur les événements indexés, et générer des réponses augmentées à partir des données présentes dans la base de données vectorielle.",
    version="1.0",
)

class AskRequest(BaseModel):
    question: str


@app.get("/health")
def root():
    return {"message": "API pour l'utilisation du Chatbot",
            "endpoints": ["/ask", "/rebuild", "/rebuild/full"]}


@app.post("/ask")
def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    try:
        answer = rag_bot.chat(payload.question)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
def rebuild():
    try:
        rag_bot.vector_store.rebuild()
        return {"status": "OK", "message": "Base de données vectorielle rechargée"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild/full")
def rebuild_full():
    try:
        builder = IndexBuilder(BASE_DIR)
        builder.run()
        rag_bot.vector_store.rebuild()
        return {"status": "OK", "message": "Rebuild complet de la BDD vectorielle terminé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",  # Écoute sur toutes les interfaces
        port=8000,       # Port par défaut
        reload=True      # Recharge automatique en développement
    )