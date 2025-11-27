"""
Script de test de l'api et des Endpoints
"""

from fastapi.testclient import TestClient
from api import app  # import de l'api FastAPI
from unittest.mock import patch


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "/ask" in response.json()["endpoints"]

print("Test de l'endpoint Health réussi ! ")

def test_ask_with_mock():
    with patch("api.rag_bot.chat", return_value="Bonjour, je suis votre assistant. Comment puis-je vous aider?"):
        payload = {"question": "Bonjour"}
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        assert response.json()["response"] == "Bonjour, je suis votre assistant. Comment puis-je vous aider?"

print("Test de l'endpoint Ask réussi ! Réponse obtenue à la question de l'utilisateur. ")


def test_rebuild():
    response = client.post("/rebuild")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert "Base de données vectorielle rechargée" in json_data["message"]

print("Test de l'endpoint Rebuild réussi ! Base de donnée vectorielle rechargée. ")
print("test terminé et réussi avec succès ! ")