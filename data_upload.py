#data_upload.py 

import json
import re
import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from html import unescape

# --------------------------------------------------------------------
# 1️⃣ Fonction de nettoyage HTML
# --------------------------------------------------------------------
def nettoyer_html(texte):
    """Supprime les balises HTML et nettoie le texte"""
    if not texte:
        return None
    
    texte = unescape(texte)
    soup = BeautifulSoup(texte, "html.parser")
    texte_propre = soup.get_text()
    texte_propre = re.sub(r'\s+', ' ', texte_propre).strip()
    return texte_propre if texte_propre else None

# --------------------------------------------------------------------
# 2️⃣ Fonction de nettoyage d’un événement
# --------------------------------------------------------------------
def nettoyer_evenement(event):
    """Nettoie un événement individuel"""

    desc = nettoyer_html(event.get("description_fr"))
    longdesc = nettoyer_html(event.get("longdescription_fr"))

    if desc and longdesc:
        description_complete = f"{desc}. {longdesc}"
    else:
        description_complete = desc or longdesc

    return {
        "uid": event.get("uid"),
        "title_fr": event.get("title_fr"),
        "description_full_fr": description_complete,  # champ fusionné
        "location_city": event.get("location_city"),
        "location_name": event.get("location_name"),
        "location_address": event.get("location_address"),
        "location_coordinates": event.get("location_coordinates"),
        "firstdate_begin": event.get("firstdate_begin"),
        "firstdate_end": event.get("firstdate_end"),
        "keywords_fr": event.get("keywords_fr")
    }

# --------------------------------------------------------------------
# 3️⃣ Téléchargement de l'API OpenAgenda via Opendatasoft
# --------------------------------------------------------------------

def telecharger_donnees_openagenda():
    """
    Télécharge les événements depuis l'API OpenAgenda filtrés par ville et date.
    Utilise la pagination via OFFSET.
    """
    URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    params = {
        "where": 'location_city:"Paris" AND firstdate_begin >= "2025-11-01T00:00:00"',
        "limit": 100,
        "offset": 0
    }

    print("📥 Téléchargement de la première page (100 événements max) ...")
    response = requests.get(URL, params=params)

    if response.status_code != 200:
        print(f"❌ Erreur API : {response.status_code}")
        return []

    data = response.json()
    tous_evenements = data.get("results", [])
    print(f"✅ {len(tous_evenements)} événements récupérés")

    return tous_evenements

#-------Partie pour Pagination if needed ----------------------------#
#si on veut faire la pagination on active ce script en markdown : 

    #tous_evenements = []

    #while True:
        #print(f"📥 Téléchargement offset {params['offset']} ...")
        #response = requests.get(URL, params=params)

        #if response.status_code != 200:
            #print(f"❌ Erreur API : {response.status_code}")
            #break

        #data = response.json()
        #evenements = data.get("results", [])

        #if not evenements:
           # break

        #tous_evenements.extend(evenements)

        # Si moins que limit, fin
        #if len(evenements) < params["limit"]:
            #break

        #params["offset"] += params["limit"]

    #print(f"✅ {len(tous_evenements)} événements trouvés")
    #return tous_evenements
#-------Fin Partie pour Pagination if needed ----------------------------#

# --------------------------------------------------------------------
# 4️⃣ Fonction principale : télécharge, nettoie et sauvegarde
# --------------------------------------------------------------------
def nettoyer_donnees_openagenda_api(fichier_sortie):
    """Télécharge, nettoie et sauvegarde les données OpenAgenda filtrées."""

    dossier = os.path.dirname(fichier_sortie)
    if dossier and not os.path.exists(dossier):
        os.makedirs(dossier, exist_ok=True)
    
    evenements = telecharger_donnees_openagenda()
    evenements_propres = [nettoyer_evenement(e) for e in evenements]

    donnees_propres = {
        "total_count": len(evenements_propres),
        "results": evenements_propres
    }

    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        json.dump(donnees_propres, f, ensure_ascii=False, indent=2)

    print(f"📁 Sauvegardé dans : {fichier_sortie}")
    return donnees_propres

# --------------------------------------------------------------------
# 5️⃣ Exemple d’utilisation
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Base = Projet 9/
    BASE_DIR = Path(__file__).parent.resolve()

    # /Projet 9/data/clean_data/
    CLEAN_DATA_DIR = BASE_DIR / "data" / "clean_data"
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fichier final
    OUTPUT_FILE = CLEAN_DATA_DIR / "data_events_clean.json" #ou Path(__file__).parent.parent / "data" / "clean_data" / "data_events_clean.json

    nettoyer_donnees_openagenda_api(OUTPUT_FILE)
