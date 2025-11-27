# test_upload.py pour tester le fichier data_upload.py
"""
Script de Test qui permet de tester : 
le téléchanrgement des données depuis l'Api open agenda
la nettoyage HTML appliqué
le nettoyage des évènements
l'enregistrement des évènements propres en format JSON

"""
import json
from pathlib import Path
import requests
from data_upload import nettoyer_html, nettoyer_evenement, telecharger_donnees_openagenda, nettoyer_donnees_openagenda_api

"""
----------------------------------------------------
⚙️ TEST DES FONCTIONS DU MODULE data_upload.py
----------------------------------------------------
"""

def test_nettoyer_html():
    print("\n=== Test nettoyer_html ===")
    html = "<p>Ceci est <b>un test</b> &amp; un <br> exemple.</p>"
    nettoyé = nettoyer_html(html)
    print("Entrée :", html)
    print("Sortie :", nettoyé)
    assert nettoyé == "Ceci est un test & un exemple."


def test_nettoyer_evenement():
    print("\n=== Test nettoyer_evenement ===")
    event_exemple = {
        "uid": "123",
        "title_fr": "Événement Test",
        "description_fr": "<p>Texte court</p>",
        "longdescription_fr": "<p>Description longue</p>",
        "location_city": "Paris",
        "location_name": "Salle X",
        "location_address": "1 rue du Test",
        "location_coordinates": [48.85, 2.35],
        "firstdate_begin": "2025-11-10",
        "firstdate_end": "2025-11-10",
        "keywords_fr": ["culture", "art"]
    }

    event_clean = nettoyer_evenement(event_exemple)
    print(json.dumps(event_clean, indent=2, ensure_ascii=False))
    assert event_clean["description_full_fr"] == "Texte court. Description longue"

def download_3_events():
    """Télécharge exactement 3 événements"""
    print("\n=== Téléchargement de 3 événements depuis l'API ===")

    URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    params = {
        "where": 'location_city:"Paris" AND firstdate_begin >= "2025-11-01T00:00:00"',
        "limit": 3,
        "offset": 0
    }

    response = requests.get(URL, params=params)
    data = response.json()
    events = data.get("results", [])

    print(f"📥 {len(events)} événements téléchargés")
    assert len(events) == 3
    return events

def test_nettoyer_donnees_openagenda_api_with_3_events():
    print("\n=== Test nettoyer_donnees_openagenda_api avec 3 événements ===")

    raw_events = download_3_events()
    cleaned_events = [nettoyer_evenement(e) for e in raw_events]

    output_file = Path("data/clean_data/test_clean_3_events.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_count": len(cleaned_events),
        "results": cleaned_events}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"📁 Fichier généré : {output_file}")
    assert output_file.exists()
    assert data["total_count"] == 3

    return output_file



# --------------------------------------------------------------
# 🚀 Lancement manuel des tests
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n================= LANCEMENT DES TESTS =================\n")

    test_nettoyer_html()
    test_nettoyer_evenement()

    # On récupère le fichier
    output_file = test_nettoyer_donnees_openagenda_api_with_3_events()

    print("\n🎉 Tous les tests se sont exécutés sans erreur !\n")