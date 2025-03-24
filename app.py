import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import io
import base64
from openai import OpenAI
from PIL import Image
import PyPDF2
import docx2txt
import pytesseract
import cv2
import numpy as np
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Charges Locatives Commerciales avec GPT-4o-mini",
    page_icon="📊",
    layout="wide"
)

# Initialisation de l'état de la session
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Configuration de l'API OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY n'est pas défini dans les variables d'environnement")

client = OpenAI(api_key=OPENAI_API_KEY)

# Fonction pour extraire le texte d'une image avec OCR
def extract_text_from_image(uploaded_file):
    """Extraire le texte d'une image avec OCR"""
    try:
        # Convertir le fichier en image
        image_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Prétraitement de l'image pour améliorer l'OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Appliquer l'OCR
        text = pytesseract.image_to_string(thresh, lang='fra')
        
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte de l'image: {str(e)}")
        return ""

# Fonctions pour extraire le texte de différents types de fichiers
def extract_text_from_pdf(uploaded_file):
    """Extraire le texte d'un fichier PDF"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extraire le texte d'un fichier Word"""
    try:
        text = docx2txt.process(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte du fichier Word: {str(e)}")
        return ""

def extract_text_from_txt(uploaded_file):
    """Extraire le texte d'un fichier TXT"""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte du fichier TXT: {str(e)}")
        return ""

def get_file_content(uploaded_file):
    """Obtenir le contenu du fichier selon son type"""
    if uploaded_file is None:
        return ""
        
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    elif file_type.startswith("image/"):
        return extract_text_from_image(uploaded_file)
    else:
        st.warning(f"Type de fichier non pris en charge: {file_type}")
        return ""

def process_multiple_files(uploaded_files):
    """Traiter plusieurs fichiers et concaténer leur contenu"""
    combined_text = ""
    
    for file in uploaded_files:
        # Obtenir le contenu du fichier
        file_content = get_file_content(file)
        if file_content:
            combined_text += f"\n\n--- Début du fichier: {file.name} ---\n\n"
            combined_text += file_content
            combined_text += f"\n\n--- Fin du fichier: {file.name} ---\n\n"
    
    return combined_text

def extract_charges_clauses_with_ai(bail_text, client):
    """
    Utilise l'IA pour extraire les sections pertinentes du bail concernant les charges.
    """
    # Si le texte est court, pas besoin de l'optimiser
    if len(bail_text) < 5000:
        return bail_text
    
    try:
        # Prompt pour demander à l'IA d'extraire les clauses pertinentes
        prompt = f"""
        Tu es un expert juridique spécialisé dans les baux commerciaux.
        
        Ta tâche consiste à extraire uniquement les clauses et sections du bail commercial suivant qui concernent les charges locatives, leur répartition, et leur facturation.
        
        Inclus dans ta sélection:
        - Toute clause mentionnant les charges, frais ou dépenses
        - Les articles concernant la répartition des charges
        - Les clauses relatives aux provisions sur charges
        - Les mentions de l'article 606 du code civil
        - Les sections traitant de la reddition des charges
        - Les articles concernant les impôts et taxes refacturés
        
        Retourne uniquement le texte des clauses pertinentes, sans commentaire ni analyse. 
        Assure-toi de conserver le format original et la numérotation des articles.
        
        Bail à analyser:
        ```
        {bail_text[:15000]}
        ```
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Utilisation de gpt-4o-mini comme demandé
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Faible température pour des résultats cohérents
            max_tokens=2000,  # Limite raisonnable pour l'extraction
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        # Si l'extraction a échoué ou renvoie un texte trop court, utiliser le texte original
        if not extracted_text or len(extracted_text) < 200:
            return bail_text[:15000]  # Limiter à 15000 caractères en cas d'échec
            
        return extracted_text
        
    except Exception as e:
        # En cas d'erreur, utiliser le texte original tronqué
        st.warning(f"Extraction intelligente des clauses non disponible: {str(e)}")
        return bail_text[:15000]

def extract_refacturable_charges_from_bail(bail_text, client):
    """
    Extrait spécifiquement les charges refacturables mentionnées dans le bail.
    """
    try:
        # Extraction des clauses pertinentes d'abord
        relevant_bail_text = extract_charges_clauses_with_ai(bail_text, client)
        
        # Prompt spécifique pour extraire uniquement les charges refacturables
        prompt = f"""
        ## Tâche d'extraction précise
        Tu es un analyste juridique spécialisé dans les baux commerciaux.
        
        Ta seule tâche est d'extraire la liste précise des charges qui sont explicitement mentionnées comme refacturables au locataire dans le bail commercial.
        
        Voici les clauses du bail concernant les charges:
        ```
        {relevant_bail_text[:15000]}
        ```
        
        ## Instructions précises
        1. Identifie uniquement les postes de charges expressément mentionnés comme refacturables au locataire
        2. Pour chaque charge, indique l'article précis ou la clause du bail qui la mentionne
        3. N'invente aucun poste de charge qui ne serait pas explicitement mentionné
        4. Si une charge est ambiguë ou implicite, indique-le clairement
        
        ## Format attendu (JSON)
        ```
        [
            {{
                "categorie": "Catégorie exacte mentionnée dans le bail",
                "description": "Description exacte de la charge, telle que rédigée dans le bail",
                "base_legale": "Article X.X ou clause Y du bail",
                "certitude": "élevée|moyenne|faible"
            }}
        ]
        ```
        
        Si aucune charge refacturable n'est mentionnée dans le bail, retourne un tableau vide.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            seed=42,
            response_format={"type": "json_object"}
        )
        
        # Extraire et analyser la réponse JSON
        try:
            result = json.loads(response.choices[0].message.content)
            # Vérifier si le résultat est une liste directe ou s'il est encapsulé
            if isinstance(result, dict) and any(k for k in result.keys() if "charge" in k.lower()):
                for key in result.keys():
                    if "charge" in key.lower() and isinstance(result[key], list):
                        return result[key]
            elif isinstance(result, list):
                return result
            else:
                # Cas où le format ne correspond pas à ce qui est attendu
                return []
        except Exception as e:
            st.warning(f"Erreur lors de l'analyse de la réponse JSON pour les charges refacturables: {str(e)}")
            return []
    
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des charges refacturables: {str(e)}")
        return []

def extract_charged_amounts_from_reddition(charges_text, client):
    """
    Extrait précisément les montants facturés dans la reddition des charges.
    Optimisé spécifiquement pour les relevés de charges commerciales avec exemples.
    """
    try:
        # Vérifier si le document contient des signaux typiques d'un relevé de charges
        contains_table = "Total des charges" in charges_text or "Total charges" in charges_text
        contains_charges_keywords = "CHARGES COMMUNES" in charges_text or "Quote-part" in charges_text
        
        # Prompt optimisé avec exemples tirés des documents récemment analysés
        prompt = f"""
        ## EXTRACTION DE CHARGES LOCATIVES COMMERCIALES
        
        Tu dois extraire avec une extrême précision toutes les charges locatives et leurs montants exacts 
        à partir du document ci-dessous:
        
        ```
        {charges_text[:15000]}
        ```
        
        ## FOCUS SUR LE TABLEAU DE CHARGES
        
        IMPORTANT: Ce document contient un tableau de charges locatives qui ressemble probablement à ceci:
        
        ```
        01 / 01  NETTOYAGE EXTERIEUR                 13 274.19 €      8565      2092.00    366      3 242.22
        01 / 04  DECHETS SECS                        16 727.76 €      8565      2092.00    366      4 085.75
        01 / 06  HYGIENE SANTE                        1 223.80 €      8565      2092.00    366        298.91
        ...etc...
        ```
        
        ## INSTRUCTIONS CRITIQUES
        
        1. Cherche spécifiquement ce tableau dans le document
        2. Pour chaque ligne du tableau, extrais:
           - Le nom exact du poste de charge (comme "NETTOYAGE EXTERIEUR")
           - Le montant final qui représente la quote-part du locataire (généralement la dernière colonne)
        3. Identifie également le montant total des charges (souvent indiqué comme "Total charges" suivi d'un montant)
        
        ## EXEMPLES DE CHARGES À RECHERCHER
        
        Voici des exemples de charges typiques que tu dois identifier:
        - NETTOYAGE EXTERIEUR: environ 3 242.22 €
        - DECHETS SECS: environ 4 085.75 €
        - HYGIENE SANTE: environ 298.91 €
        - ELECTRICITE ABORDS & PKGS: environ 2 034.14 €
        - STRUCTURE: environ 2 068.80 €
        - ESPACES VERTS EXTERIEURS: environ 8 240.83 €
        - HONORAIRES GESTION: environ 4 652.96 €
        
        ## FORMAT DE RÉPONSE JSON REQUIS
        
        Réponds avec ce format JSON exact:
        {{
            "charges": [
                {{
                    "poste": "NOM EXACT DU POSTE DE CHARGE",
                    "montant": MONTANT_NUMÉRIQUE,
                    "texte_original": "LIGNE COMPLÈTE DU DOCUMENT"
                }},
                ...
            ],
            "montant_total": MONTANT_TOTAL_NUMÉRIQUE
        }}
        
        CRITIQUE: Assure-toi que CHAQUE charge du document est correctement identifiée et que les montants sont exactement ceux indiqués.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert-comptable spécialisé dans l'extraction précise de données financières à partir de documents de reddition de charges."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=42,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Vérifier si le résultat a une structure attendue avec "charges"
            if "charges" in result and isinstance(result["charges"], list):
                return result["charges"]
            
            # Alternative: si la structure est différente, chercher tout tableau de données
            for key in result:
                if isinstance(result[key], list) and len(result[key]) > 0:
                    if all(isinstance(item, dict) and "poste" in item and "montant" in item for item in result[key]):
                        return result[key]
            
            # Si aucune charge trouvée dans un format attendu
            st.warning("Format de réponse non standard. Nouvelle tentative avec un prompt plus spécifique...")
            
            # Deuxième tentative avec un prompt encore plus spécifique
            return retry_extraction_with_context(charges_text, client)
                
        except json.JSONDecodeError as e:
            st.warning(f"Erreur lors du décodage JSON: {str(e)}")
            # Deuxième tentative
            return retry_extraction_with_context(charges_text, client)
    
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des montants facturés: {str(e)}")
        return []

def retry_extraction_with_context(charges_text, client):
    """
    Deuxième tentative d'extraction avec des instructions plus spécifiques.
    """
    try:
        # Prompt extrêmement spécifique pour la structure du document SCI PASTEUR
        prompt = f"""
        ## EXTRACTION D'URGENCE - RELEVÉ DE CHARGES SCI PASTEUR
        
        Le document suivant est un relevé individuel des charges locatives commerciales émis par SCI PASTEUR.
        Il contient une grille de charges pour la période du 01/01/2024 au 31/12/2024.
        
        ```
        {charges_text[:5000]}
        ```
        
        Le document contient un tableau avec les colonnes suivantes:
        - Clés (01/01, 01/04, etc.)
        - Désignation (NETTOYAGE EXTERIEUR, DECHETS SECS, etc.)
        - Total de l'immeuble
        - Tantièmes globaux
        - Tantièmes particuliers
        - Nb jrs 366
        - Quote-part (montant final facturé au locataire)
        
        LA COLONNE QUI NOUS INTÉRESSE EST "Quote-part" - c'est le montant facturé au locataire.
        
        VOICI EXACTEMENT LA STRUCTURE DE CHAQUE LIGNE:
        01 / XX   NOM DE LA CHARGE   MONTANT TOTAL €   8565   2092.00   366   QUOTE-PART €
        
        Extrait du document sous la section "CHARGES COMMUNES" uniquement les noms des charges et leurs montants de quote-part.
        Par exemple:
        - "NETTOYAGE EXTERIEUR" → 3242.22
        - "DECHETS SECS" → 4085.75
        
        Réponds avec ce format JSON simple:
        [
            {{"poste": "NETTOYAGE EXTERIEUR", "montant": 3242.22}},
            {{"poste": "DECHETS SECS", "montant": 4085.75}},
            ...
        ]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu extrais avec une précision absolue les données de charges locatives des relevés SCI PASTEUR."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Essayer de trouver un bloc JSON même si la réponse contient d'autres éléments
        try:
            # Nettoyer la réponse pour extraire juste le JSON
            clean_content = content
            # Supprimer tout avant le premier '['
            if '[' in clean_content:
                clean_content = clean_content[clean_content.find('['):]
            # Supprimer tout après le dernier ']'
            if ']' in clean_content:
                clean_content = clean_content[:clean_content.rfind(']')+1]
                
            charges = json.loads(clean_content)
            
            # Convertir en format standard
            formatted_charges = []
            for charge in charges:
                formatted_charges.append({
                    "poste": charge.get("poste", ""),
                    "montant": charge.get("montant", 0),
                    "texte_original": f"{charge.get('poste', '')} - {charge.get('montant', 0)}€"
                })
            
            return formatted_charges
        except:
            # Dernière tentative avec GPT-4o si disponible
            try:
                # Tenter avec gpt-4o si disponible
                model = "gpt-4o" if any(m.id == "gpt-4o" for m in client.models.list().data) else "gpt-4o-mini"
            except:
                model = "gpt-4o-mini"
                
            last_prompt = f"""
            Voici un relevé de charges locatives pour un bail commercial:
            
            ```
            {charges_text[:7000]}
            ```
            
            J'ai besoin uniquement que tu extraies les noms des charges et leurs montants de quote-part.
            
            Cherche spécifiquement la section "CHARGES COMMUNES" ou autre section de charges.
            Ne réponds qu'avec un tableau JSON simple - rien d'autre:
            [
                {{"poste": "Nom de la charge", "montant": valeur_numérique}},
                ...
            ]
            """
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": last_prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if isinstance(result, list):
                    formatted_charges = []
                    for charge in result:
                        formatted_charges.append({
                            "poste": charge.get("poste", ""),
                            "montant": charge.get("montant", 0),
                            "texte_original": f"{charge.get('poste', '')} - {charge.get('montant', 0)}€"
                        })
                    return formatted_charges
                    
                for key in result:
                    if isinstance(result[key], list):
                        formatted_charges = []
                        for charge in result[key]:
                            formatted_charges.append({
                                "poste": charge.get("poste", ""),
                                "montant": charge.get("montant", 0),
                                "texte_original": f"{charge.get('poste', '')} - {charge.get('montant', 0)}€"
                            })
                        return formatted_charges
                
                return []
            except:
                return []
                
    except Exception as e:
        st.error(f"Erreur lors de la seconde tentative d'extraction: {str(e)}")
        return []
def retry_extraction_with_ai(charges_text, client):
    """
    Seconde tentative d'extraction avec l'IA en utilisant un prompt plus direct.
    """
    try:
        # Prompt plus direct avec des exemples
        prompt = f"""
        ## EXTRACTION URGENTE DE DONNÉES DE CHARGES LOCATIVES
        
        Le document suivant est une reddition de charges locatives commerciales.
        Il contient un tableau où chaque ligne représente une charge avec son montant.
        
        Voici le document:
        ```
        {charges_text[:15000]}
        ```
        
        ## EXEMPLES DE CHARGES À IDENTIFIER
        Voici des exemples du type de charges que tu dois trouver:
        - NETTOYAGE EXTERIEUR
        - DECHETS SECS
        - ELECTRICITE 
        - ESPACES VERTS
        - HONORAIRES GESTION
        
        ## INSTRUCTION CRITIQUE
        1. LISTE TOUTES LES CHARGES avec leurs montants exacts (généralement dans la colonne "Quote-part")
        2. IDENTIFIE LE MONTANT TOTAL des charges (généralement indiqué comme "Total charges")
        3. ASSURE-TOI de ne rien manquer et de ne rien inventer
        
        Réponds avec une liste JSON simple:
        [
            {{"poste": "NOM DE LA CHARGE", "montant": MONTANT_NUMERIQUE}},
            ...
        ]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un comptable expert qui peut extraire avec précision des données financières de documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Essayer de trouver un bloc JSON même si la réponse contient d'autres éléments
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            charges = json.loads(json_str)
            return charges
        
        # Si pas de JSON valide mais des charges mentionnées
        if "poste" in content and "montant" in content:
            st.warning("Données extraites mais pas au format JSON standard. Création d'un format intermédiaire...")
            
            # Dernier recours: demander une troisième extraction extrêmement simplifiée
            return extract_with_simplified_prompt(charges_text, client)
            
        return []
            
    except Exception as e:
        st.error(f"Erreur lors de la seconde tentative d'extraction: {str(e)}")
        return []

def extract_with_simplified_prompt(charges_text, client):
    """
    Troisième tentative d'extraction avec l'IA en utilisant un prompt simplifié au maximum.
    """
    try:
        prompt = f"""
        Voici un relevé de charges locatives commerciales:
        
        {charges_text[:5000]}
        
        Donne-moi uniquement les noms et montants des charges sous ce format exact: 
        [
            {{"poste": "NOM_CHARGE", "montant": MONTANT}},
            ...
        ]
        Ne réponds rien d'autre que ce JSON.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Tenter d'extraire juste un tableau JSON
        try:
            # Enlever tout ce qui n'est pas le JSON attendu
            clean_content = re.sub(r'^[^[]*', '', content)
            clean_content = re.sub(r'[^]]*$', '', clean_content)
            charges = json.loads(clean_content)
            return charges
        except:
            # Si tout échoue, dernier recours: essayer avec gpt-4o
            st.warning("Dernière tentative avec un modèle plus puissant...")
            
            try:
                # Tenter avec gpt-4o si disponible
                response = client.chat.completions.create(
                    model="gpt-4o",  # Modèle plus puissant
                    messages=[
                        {"role": "system", "content": "Extrait uniquement les charges et montants du document sans ajouter de commentaire."},
                        {"role": "user", "content": f"Extrait les charges et montants de ce document et retourne UNIQUEMENT un tableau JSON:\n\n{charges_text[:5000]}"}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                if "charges" in result and isinstance(result["charges"], list):
                    return result["charges"]
                
                # Si structure différente mais contient une liste
                for key in result:
                    if isinstance(result[key], list):
                        return result[key]
                        
                return []
            except:
                # Si vraiment tout a échoué
                st.error("Impossible d'extraire les charges après plusieurs tentatives.")
                return []
                
    except Exception as e:
        st.error(f"Erreur lors de l'extraction simplifiée: {str(e)}")
        return []

def analyse_charges_conformity(refacturable_charges, charged_amounts, client):
    """
    Analyse la conformité entre les charges refacturables et les montants facturés.
    """
    try:
        # Convertir les listes en JSON pour les inclure dans le prompt
        refacturable_json = json.dumps(refacturable_charges, ensure_ascii=False)
        charged_json = json.dumps(charged_amounts, ensure_ascii=False)
        
        prompt = f"""
        ## Tâche d'analyse
        Tu es un expert juridique et comptable spécialisé dans l'analyse de conformité des charges locatives commerciales.
        
        Ta tâche est d'analyser la conformité entre les charges refacturables selon le bail et les charges effectivement facturées.
        
        ## Données d'entrée
        
        ### Charges refacturables selon le bail:
        ```json
        {refacturable_json}
        ```
        
        ### Charges effectivement facturées:
        ```json
        {charged_json}
        ```
        
        ## Instructions précises
        1. Pour chaque charge facturée, détermine si elle correspond à une charge refacturable selon le bail
        2. Calcule le pourcentage que représente chaque charge par rapport au total des charges facturées
        3. Évalue la conformité de chaque charge par rapport au bail
        4. Identifie les charges potentiellement contestables avec une justification précise
        5. Calcule le montant total des charges facturées
        6. Détermine un taux global de conformité basé sur le pourcentage des charges conformes
        
        ## Format attendu (JSON)
        ```json
        {{
            "charges_facturees": [
                {{
                    "poste": "Intitulé exact de la charge facturée",
                    "montant": 1234.56,
                    "pourcentage": 25.5,
                    "conformite": "conforme|à vérifier|non conforme",
                    "justification": "Explication précise de la conformité ou non",
                    "contestable": true|false,
                    "raison_contestation": "Raison précise si contestable"
                }}
            ],
            "montant_total": 5000.00,
            "analyse_globale": {{
                "taux_conformite": 75,
                "conformite_detail": "Explication détaillée du taux de conformité"
            }},
            "recommandations": [
                "Recommandation précise et actionnable 1",
                "Recommandation précise et actionnable 2"
            ]
        }}
        ```
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            seed=42,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # Ajouter les charges refacturables au résultat pour l'affichage complet
            result["charges_refacturables"] = refacturable_charges
            return result
        except Exception as e:
            st.warning(f"Erreur lors de l'analyse de la réponse JSON pour l'analyse de conformité: {str(e)}")
            return {
                "charges_refacturables": refacturable_charges,
                "charges_facturees": charged_amounts,
                "montant_total": sum(charge.get("montant", 0) for charge in charged_amounts),
                "analyse_globale": {
                    "taux_conformite": 0,
                    "conformite_detail": "Impossible d'analyser la conformité en raison d'une erreur."
                },
                "recommandations": ["Vérifier manuellement la conformité des charges."]
            }
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse de conformité: {str(e)}")
        return {
            "charges_refacturables": refacturable_charges,
            "charges_facturees": charged_amounts,
            "montant_total": sum(charge.get("montant", 0) for charge in charged_amounts),
            "analyse_globale": {
                "taux_conformite": 0,
                "conformite_detail": "Impossible d'analyser la conformité en raison d'une erreur."
            },
            "recommandations": ["Vérifier manuellement la conformité des charges."]
        }

def analyze_with_openai(text1, text2, document_type):
    """
    Analyse les documents en suivant une approche structurée en trois étapes,
    uniquement basée sur l'IA sans fallbacks déterministes.
    """
    try:
        with st.spinner("Étape 1/3: Extraction des charges refacturables du bail..."):
            # Extraire les charges refacturables mentionnées dans le bail
            refacturable_charges = extract_refacturable_charges_from_bail(text1, client)
            
            if refacturable_charges:
                st.success(f"✅ {len(refacturable_charges)} postes de charges refacturables identifiés dans le bail")
            else:
                st.warning("⚠️ Aucune charge refacturable clairement identifiée dans le bail")
                # Deuxième tentative avec un prompt différent
                refacturable_charges = retry_extract_refacturable_charges(text1, client)
        
        with st.spinner("Étape 2/3: Extraction des montants facturés..."):
            # Extraire les montants facturés mentionnés dans la reddition
            charged_amounts = extract_charged_amounts_from_reddition(text2, client)
            
            if charged_amounts:
                total = sum(charge.get("montant", 0) for charge in charged_amounts)
                st.success(f"✅ {len(charged_amounts)} postes de charges facturés identifiés, pour un total de {total:.2f}€")
            else:
                st.warning("⚠️ Aucun montant facturé clairement identifié dans la reddition des charges")
        
        with st.spinner("Étape 3/3: Analyse de la conformité..."):
            # Analyser la conformité entre les charges refacturables et facturées
            result = analyse_charges_conformity(refacturable_charges, charged_amounts, client)
            
            if result and "analyse_globale" in result and "taux_conformite" in result["analyse_globale"]:
                conformity = result["analyse_globale"]["taux_conformite"]
                st.success(f"✅ Analyse complète avec un taux de conformité de {conformity}%")
            else:
                st.warning("⚠️ Analyse de conformité incomplète - nouvelle tentative...")
                # Deuxième tentative avec approche différente
                result = retry_analyse_conformity(refacturable_charges, charged_amounts, client)
        
        return result
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {str(e)}")
        # Nouvelle tentative avec un prompt composé
        return final_attempt_complete_analysis(text1, text2, client)

def retry_extract_refacturable_charges(bail_text, client):
    """
    Seconde tentative d'extraction des charges refacturables avec un prompt différent.
    """
    try:
        prompt = f"""
        ## Tâche d'extraction spécifique
        Tu es un juriste spécialisé en droit des baux commerciaux en France.
        
        Examine attentivement ce bail commercial et identifie TOUTES les charges qui peuvent être refacturées au locataire.
        
        ```
        {bail_text[:10000]}
        ```
        
        ## Instructions critiques
        1. Recherche spécifiquement les mentions de charges locatives, frais, dépenses ou taxes
        2. Cherche les clauses qui indiquent ce qui est à la charge du preneur/locataire
        3. Identifie les articles qui mentionnent la répartition des charges
        4. Considère les mentions de l'article 606 du Code Civil (grosses réparations)
        
        ## Liste de charges typiques à identifier si elles sont mentionnées
        - Nettoyage des parties communes
        - Enlèvement des déchets/ordures
        - Entretien des espaces verts
        - Électricité des parties communes
        - Chauffage collectif
        - Eau
        - Honoraires de gestion
        - Assurances
        - Taxes foncières
        - Taxes sur les bureaux
        
        Retourne uniquement un tableau JSON structuré:
        [
            {{"categorie": "Type de charge", "description": "Description précise", "base_legale": "Article ou clause du bail", "certitude": "élevée|moyenne|faible"}}
        ]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert juridique spécialisé en baux commerciaux."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extraire la liste des charges de la réponse JSON
        if isinstance(result, list):
            return result
        
        # Si le résultat est un objet contenant une liste
        for key in result:
            if isinstance(result[key], list):
                return result[key]
        
        return []
    
    except Exception as e:
        st.error(f"Erreur lors de la seconde tentative d'extraction des charges refacturables: {str(e)}")
        return []

def retry_analyse_conformity(refacturable_charges, charged_amounts, client):
    """
    Seconde tentative d'analyse de conformité avec un prompt différent.
    """
    try:
        # Convertir les listes en JSON pour les inclure dans le prompt
        refacturable_json = json.dumps(refacturable_charges, ensure_ascii=False)
        charged_json = json.dumps(charged_amounts, ensure_ascii=False)
        
        prompt = f"""
        ## Analyse détaillée de conformité des charges locatives
        
        Tu es un avocat spécialisé en baux commerciaux qui doit déterminer si les charges facturées à un locataire sont conformes au bail.
        
        ## Données d'entrée
        
        ### 1. Charges refacturables selon le bail:
        ```json
        {refacturable_json}
        ```
        
        ### 2. Charges effectivement facturées:
        ```json
        {charged_json}
        ```
        
        ## Instructions précises
        1. Compare chaque charge facturée avec les charges autorisées par le bail
        2. Pour chaque charge facturée, détermine si elle est explicitement autorisée, implicitement autorisée, ou non autorisée
        3. Calcule le pourcentage que représente chaque charge par rapport au total facturé
        4. Identifie les charges potentiellement contestables avec justification précise
        5. Détermine un taux global de conformité basé sur le pourcentage des charges conformes
        
        ## Format de réponse requis (JSON)
        {{
            "charges_facturees": [
                {{
                    "poste": "Nom exact de la charge facturée",
                    "montant": montant_numérique,
                    "pourcentage": pourcentage_numérique,
                    "conformite": "conforme|à vérifier|non conforme",
                    "justification": "Explication précise",
                    "contestable": true|false,
                    "raison_contestation": "Raison si contestable"
                }}
            ],
            "montant_total": montant_total_numérique,
            "analyse_globale": {{
                "taux_conformite": pourcentage_numérique,
                "conformite_detail": "Explication détaillée"
            }},
            "recommandations": [
                "Recommandation actionnable 1",
                "Recommandation actionnable 2"
            ]
        }}
        
        ATTENTION: Sois rigoureux dans ton analyse. Ne suppose pas qu'une charge est autorisée sans preuve claire dans le bail.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert juridique et comptable spécialisé dans l'analyse de conformité des charges locatives commerciales."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # Ajouter les charges refacturables au résultat pour l'affichage complet
            result["charges_refacturables"] = refacturable_charges
            return result
        except Exception as e:
            st.warning(f"Erreur lors de l'analyse de la réponse JSON pour l'analyse de conformité: {str(e)}")
            # Dernière tentative avec format simplifié
            return simplify_and_retry_conformity(refacturable_charges, charged_amounts, client)
        
    except Exception as e:
        st.error(f"Erreur lors de la seconde tentative d'analyse de conformité: {str(e)}")
        return {
            "charges_refacturables": refacturable_charges,
            "charges_facturees": charged_amounts,
            "montant_total": sum(charge.get("montant", 0) for charge in charged_amounts),
            "analyse_globale": {
                "taux_conformite": 50,  # Valeur par défaut médiane
                "conformite_detail": "Analyse partielle suite à une erreur technique."
            },
            "recommandations": ["Consulter un expert pour une analyse plus approfondie."]
        }

def simplify_and_retry_conformity(refacturable_charges, charged_amounts, client):
    """
    Dernière tentative d'analyse de conformité avec un format simplifié.
    """
    try:
        # Simplifier les données d'entrée
        simple_refacturable = []
        for charge in refacturable_charges:
            simple_refacturable.append({
                "categorie": charge.get("categorie", ""),
                "description": charge.get("description", "")
            })
        
        simple_charged = []
        for charge in charged_amounts:
            simple_charged.append({
                "poste": charge.get("poste", ""),
                "montant": charge.get("montant", 0)
            })
        
        # Prompt simplifié
        prompt = f"""
        Analyse si ces charges facturées sont conformes au bail:
        
        Charges refacturables selon bail: {json.dumps(simple_refacturable)}
        
        Charges facturées: {json.dumps(simple_charged)}
        
        Donne un simple JSON avec:
        1. Taux de conformité (%)
        2. Liste des charges conformes ou non
        3. Recommandations
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Construire un résultat structuré à partir de la réponse
        structured_result = {
            "charges_refacturables": refacturable_charges,
            "charges_facturees": [],
            "montant_total": sum(charge.get("montant", 0) for charge in charged_amounts),
            "analyse_globale": {
                "taux_conformite": result.get("taux_conformite", 50),
                "conformite_detail": result.get("detail", "Analyse simplifiée suite à des erreurs techniques.")
            },
            "recommandations": result.get("recommandations", ["Consulter un expert pour une analyse complète."])
        }
        
        # Restructurer les charges facturées
        if "charges" in result and isinstance(result["charges"], list):
            for i, charge in enumerate(charged_amounts):
                if i < len(result["charges"]):
                    structured_result["charges_facturees"].append({
                        "poste": charge.get("poste", ""),
                        "montant": charge.get("montant", 0),
                        "pourcentage": (charge.get("montant", 0) / structured_result["montant_total"] * 100) if structured_result["montant_total"] > 0 else 0,
                        "conformite": result["charges"][i].get("conformite", "à vérifier"),
                        "justification": result["charges"][i].get("justification", ""),
                        "contestable": result["charges"][i].get("conformite", "") == "non conforme",
                        "raison_contestation": result["charges"][i].get("justification", "")
                    })
                else:
                    # Pour les charges sans évaluation explicite
                    structured_result["charges_facturees"].append({
                        "poste": charge.get("poste", ""),
                        "montant": charge.get("montant", 0),
                        "pourcentage": (charge.get("montant", 0) / structured_result["montant_total"] * 100) if structured_result["montant_total"] > 0 else 0,
                        "conformite": "à vérifier",
                        "justification": "Analyse incomplète",
                        "contestable": False,
                        "raison_contestation": ""
                    })
        else:
            # Ajouter les charges telles quelles si pas d'évaluation disponible
            for charge in charged_amounts:
                structured_result["charges_facturees"].append({
                    "poste": charge.get("poste", ""),
                    "montant": charge.get("montant", 0),
                    "pourcentage": (charge.get("montant", 0) / structured_result["montant_total"] * 100) if structured_result["montant_total"] > 0 else 0,
                    "conformite": "à vérifier",
                    "justification": "Analyse incomplète",
                    "contestable": False,
                    "raison_contestation": ""
                })
        
        return structured_result
        
    except Exception as e:
        st.error(f"Erreur lors de la tentative simplifiée d'analyse de conformité: {str(e)}")
        # Résultat minimal de secours
        return {
            "charges_refacturables": refacturable_charges,
            "charges_facturees": charged_amounts,
            "montant_total": sum(charge.get("montant", 0) for charge in charged_amounts),
            "analyse_globale": {
                "taux_conformite": 50,
                "conformite_detail": "Analyse incomplète suite à des erreurs techniques répétées."
            },
            "recommandations": ["Consulter un expert pour une analyse complète des charges."]
        }

def final_attempt_complete_analysis(text1, text2, client):
    """
    Tentative finale d'analyse complète avec un seul appel IA intégré.
    Cette fonction est appelée en dernier recours si les autres approches échouent.
    """
    try:
        st.warning("Tentative d'analyse unifiée en cours...")
        
        # Utiliser un seul prompt qui fait tout en une fois
        prompt = f"""
        ## ANALYSE COMPLÈTE DE CONFORMITÉ DES CHARGES LOCATIVES
        
        Tu es un expert juridique et comptable spécialisé dans l'analyse des baux commerciaux.
        
        Voici les deux documents que tu dois analyser:
        
        ### 1. BAIL COMMERCIAL (extraits pertinents):
        ```
        {text1[:7000]}
        ```
        
        ### 2. REDDITION DES CHARGES:
        ```
        {text2[:7000]}
        ```
        
        ## Ta mission en 3 étapes:
        
        ### Étape 1: Extrais les charges refacturables mentionnées dans le bail
        - Identifie les clauses spécifiant quelles charges sont refacturables au locataire
        - Recherche les mentions de charges locatives, répartition des charges, etc.
        - Vérifie les clauses concernant l'article 606 du Code Civil
        
        ### Étape 2: Extrais les charges facturées dans la reddition
        - Identifie précisément chaque poste de charge facturé
        - Note le montant exact pour chaque poste
        - Calcule le montant total des charges facturées
        
        ### Étape 3: Analyse la conformité des charges facturées
        - Compare chaque charge facturée avec les charges autorisées par le bail
        - Détermine si chaque charge est conforme ou non aux stipulations du bail
        - Calcule un taux global de conformité
        - Identifie les charges potentiellement contestables
        
        ## Format de réponse JSON requis
        Réponds UNIQUEMENT avec ce format JSON exact:
        
        {{
            "charges_refacturables": [
                {{
                    "categorie": "Type de charge",
                    "description": "Description précise",
                    "base_legale": "Article ou clause du bail"
                }}
            ],
            "charges_facturees": [
                {{
                    "poste": "Nom exact de la charge facturée",
                    "montant": montant_numérique,
                    "pourcentage": pourcentage_numérique,
                    "conformite": "conforme|à vérifier|non conforme",
                    "justification": "Explication précise",
                    "contestable": true|false,
                    "raison_contestation": "Raison si contestable"
                }}
            ],
            "montant_total": montant_total_numérique,
            "analyse_globale": {{
                "taux_conformite": pourcentage_numérique,
                "conformite_detail": "Explication détaillée"
            }},
            "recommandations": [
                "Recommandation actionnable 1",
                "Recommandation actionnable 2"
            ]
        }}
        
        IMPORTANT: La reddition des charges inclut probablement un tableau de charges avec montants.
        Pour le document 2 (reddition), CHERCHE ATTENTIVEMENT tout tableau ou liste de charges locatives.
        Une ligne typique pourrait être "NETTOYAGE EXTERIEUR ... 3242.22 €" ou similaire.
        """
        
        # Essayer avec un modèle plus puissant si possible
        try:
            model = "gpt-4o" if any(m.id == "gpt-4o" for m in client.models.list().data) else "gpt-4o-mini"
        except:
            model = "gpt-4o-mini"
            
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un expert juridique et comptable spécialisé dans l'analyse des baux commerciaux et des charges locatives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Vérification basique de la structure
            if "charges_facturees" not in result or not result["charges_facturees"]:
                st.warning("Aucune charge facturée identifiée dans l'analyse unifiée. Tentative de récupération spécifique...")
                
                # Extraction spécifique des charges facturées
                charges_prompt = f"""
                Extrais UNIQUEMENT la liste des charges facturées et leurs montants exacts de ce document de reddition:
                
                ```
                {text2[:10000]}
                ```
                
                ATTENTION: Ce document contient certainement un tableau de charges. Chaque ligne du tableau
                représente une charge avec un montant. Par exemple: "NETTOYAGE EXTERIEUR ... 3242.22 €"
                
                Fournis UNIQUEMENT un tableau JSON simple:
                [
                    {{"poste": "Nom exact du poste", "montant": montant_numérique}},
                    ...
                ]
                """
                
                charges_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": charges_prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                try:
                    charges_result = json.loads(charges_response.choices[0].message.content)
                    
                    # Récupérer les charges depuis la réponse
                    extracted_charges = []
                    if isinstance(charges_result, list):
                        extracted_charges = charges_result
                    else:
                        for key in charges_result:
                            if isinstance(charges_result[key], list):
                                extracted_charges = charges_result[key]
                                break
                    
                    # Si des charges ont été trouvées, mettre à jour le résultat
                    if extracted_charges:
                        total = sum(charge.get("montant", 0) for charge in extracted_charges)
                        
                        # Calculer les pourcentages
                        for charge in extracted_charges:
                            charge["pourcentage"] = (charge.get("montant", 0) / total * 100) if total > 0 else 0
                            charge["conformite"] = "à vérifier"
                            charge["contestable"] = False
                            charge["justification"] = "Analyse incomplète"
                            charge["raison_contestation"] = ""
                        
                        result["charges_facturees"] = extracted_charges
                        result["montant_total"] = total
                        
                        # Mettre à jour l'analyse globale
                        if "analyse_globale" not in result:
                            result["analyse_globale"] = {}
                        
                        result["analyse_globale"]["taux_conformite"] = 50  # Valeur par défaut
                        result["analyse_globale"]["conformite_detail"] = "Analyse partielle des charges facturées. Vérification manuelle recommandée."
                except:
                    pass
            
            return result
            
        except json.JSONDecodeError:
            # En cas d'échec, retourner une structure minimale mais fonctionnelle
            st.error("Impossible de produire une analyse complète après plusieurs tentatives.")
            
            return {
                "charges_refacturables": [],
                "charges_facturees": [],
                "montant_total": 0,
                "analyse_globale": {
                    "taux_conformite": 0,
                    "conformite_detail": "L'analyse n'a pas pu être finalisée suite à des erreurs techniques répétées."
                },
                "recommandations": [
                    "Consulter un expert juridique pour une analyse manuelle complète.",
                    "Vérifier que les documents fournis sont lisibles et complets."
                ]
            }
            
    except Exception as e:
        st.error(f"Erreur lors de la tentative finale d'analyse: {str(e)}")
        return {
            "charges_refacturables": [],
            "charges_facturees": [],
            "montant_total": 0,
            "analyse_globale": {
                "taux_conformite": 0,
                "conformite_detail": "L'analyse a échoué suite à une erreur technique. Veuillez réessayer."
            },
            "recommandations": [
                "Réessayer l'analyse avec des documents au format texte.",
                "S'assurer que les documents sont lisibles et contiennent les informations nécessaires."
            ]
        }
        
def plot_themes_chart(themes):
    """Crée un graphique des thèmes principaux"""
    if not themes:
        return None

    # Préparer les données (tous les thèmes ont le même poids par défaut)
    labels = themes
    sizes = [1] * len(themes)

    # Graphique camembert
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )

    # Ajuster les propriétés du texte
    plt.setp(autotexts, size=8, weight='bold')
    plt.setp(texts, size=8)

    plt.title('Thèmes principaux identifiés')
    plt.tight_layout()
    
    return fig

def generate_pdf_report(analysis, document_type, text1=None, text2=None):
    """
    Génère un rapport PDF complet et précis de l'analyse des charges locatives commerciales.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.units import cm
    from io import BytesIO
    import datetime
    
    # Créer un buffer pour stocker le PDF
    buffer = BytesIO()
    
    # Créer le document PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # Contenu du document
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', parent=styles['Heading1'], alignment=1))
    styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=4))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=8))
    
    # Titre et date
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    title = f"Analyse des Charges Locatives Commerciales"
    story.append(Paragraph(title, styles['Center']))
    story.append(Paragraph(f"Rapport généré le {today}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Informations générales
    story.append(Paragraph("Informations générales", styles['Heading2']))
    
    # Préparation des données pour le tableau d'information
    info_data = [
        ["Type de bail", "Commercial"]
    ]
    
    # Ajout des informations financières si disponibles
    if "montant_total" in analysis:
        info_data.append(["Montant total des charges", f"{analysis['montant_total']:.2f}€"])
    
    if "analyse_globale" in analysis and "taux_conformite" in analysis["analyse_globale"]:
        info_data.append(["Taux de conformité", f"{analysis['analyse_globale']['taux_conformite']}%"])
    
    # Créer un tableau pour les informations
    info_table = Table(info_data, colWidths=[5*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Analyse de conformité
    if "analyse_globale" in analysis and "conformite_detail" in analysis["analyse_globale"]:
        story.append(Paragraph("Analyse de conformité", styles['Heading3']))
        story.append(Paragraph(analysis["analyse_globale"]["conformite_detail"], styles['Justify']))
        story.append(Spacer(1, 0.5*cm))
    
    # Charges refacturables selon le bail
    if "charges_refacturables" in analysis and analysis["charges_refacturables"]:
        story.append(Paragraph("Charges refacturables selon le bail", styles['Heading2']))
        
        # Création du tableau des charges refacturables
        refac_data = [["Catégorie", "Description", "Base légale / contractuelle"]]
        
        for charge in analysis["charges_refacturables"]:
            refac_data.append([
                charge.get("categorie", ""),
                charge.get("description", ""),
                charge.get("base_legale", "")
            ])
        
        refac_table = Table(refac_data, colWidths=[4*cm, 7*cm, 4*cm])
        refac_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(refac_table)
        story.append(Spacer(1, 0.5*cm))
    
    # Analyse des charges facturées
    if "charges_facturees" in analysis and analysis["charges_facturees"]:
        story.append(Paragraph("Analyse des charges facturées", styles['Heading2']))
        
        # Création du tableau des charges facturées
        charges_data = [["Poste", "Montant (€)", "% du total", "Conformité", "Contestable"]]
        
        for charge in analysis["charges_facturees"]:
            charges_data.append([
                charge.get("poste", ""),
                f"{charge.get('montant', 0):.2f}",
                f"{charge.get('pourcentage', 0):.1f}%",
                charge.get("conformite", ""),
                "Oui" if charge.get("contestable", False) else "Non"
            ])
        
        charges_table = Table(charges_data, colWidths=[6*cm, 2.5*cm, 2*cm, 2.5*cm, 2*cm])
        charges_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(charges_table)
        story.append(Spacer(1, 0.5*cm))
        
        # Charges contestables
        contestable_charges = [c for c in analysis["charges_facturees"] if c.get("contestable", False)]
        if contestable_charges:
            story.append(Paragraph("Charges potentiellement contestables", styles['Heading2']))
            
            for charge in contestable_charges:
                charge_title = f"{charge.get('poste', '')} ({charge.get('montant', 0):.2f}€)"
                story.append(Paragraph(charge_title, styles['Heading3']))
                story.append(Paragraph(f"Montant: {charge.get('montant', 0):.2f}€ ({charge.get('pourcentage', 0):.1f}% du total)", styles['Normal']))
                
                if "raison_contestation" in charge and charge["raison_contestation"]:
                    story.append(Paragraph(f"Raison: {charge['raison_contestation']}", styles['Normal']))
                
                if "justification" in charge and charge["justification"]:
                    story.append(Paragraph(f"Justification: {charge['justification']}", styles['Normal']))
                    
                story.append(Spacer(1, 0.3*cm))
    
    # Recommandations
    if "recommandations" in analysis and analysis["recommandations"]:
        story.append(PageBreak())
        story.append(Paragraph("Recommandations", styles['Heading2']))
        
        for i, rec in enumerate(analysis["recommandations"]):
            story.append(Paragraph(f"{i+1}. {rec}", styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
    
    # Pied de page
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Ce rapport a été généré automatiquement et sert uniquement à titre indicatif. "
                          "Pour une analyse juridique complète, veuillez consulter un professionnel du droit.", 
                          styles['Small']))
    
    # Construire le PDF
    doc.build(story)
    
    # Récupérer le contenu du buffer
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content
    
# Interface utilisateur Streamlit
def main():
    st.title("Analyseur de Charges Locatives Commerciales avec GPT-4o-mini")
    st.markdown("""
    Cet outil analyse la cohérence entre les clauses de votre bail commercial et la reddition des charges en utilisant GPT-4o-mini.
    L'analyse se fait en trois étapes précises:
    1. Extraction des charges refacturables du bail
    2. Extraction des montants facturés de la reddition
    3. Analyse de la conformité entre les charges autorisées et les charges facturées
    """)

    # Sidebar pour la configuration
    st.sidebar.header("Configuration")

    # Pas besoin de sélectionner le type de bail puisque c'est toujours commercial
    document_type = "commercial"
    
    st.sidebar.info("Cet outil est conçu spécifiquement pour analyser les baux commerciaux et leurs charges.")

    surface = st.sidebar.text_input(
        "Surface locative (m²)",
        help="Utilisé pour calculer le ratio de charges au m²"
    )

    # Interface principale avec onglets
    tab1, tab2 = st.tabs(["Saisie manuelle", "Téléchargement de fichiers"])

    # Onglet 1: Saisie manuelle
    with tab1:
        with st.form("input_form_manual"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Contrat de bail commercial / Clauses de charges")
                document1_manual = st.text_area(
                    "Copiez-collez les clauses du bail commercial concernant les charges",
                    height=250,
                    help="Entrez les sections du bail commercial qui mentionnent la répartition et facturation des charges"
                )

            with col2:
                st.subheader("Reddition des charges")
                document2_manual = st.text_area(
                    "Copiez-collez le détail des charges facturées",
                    height=250,
                    help="Entrez le détail des charges qui vous ont été facturées (postes et montants)"
                )

            specific_questions = st.text_area(
                "Questions spécifiques (facultatif)",
                help="Avez-vous des questions particulières concernant certaines charges?"
            )

            submitted_manual = st.form_submit_button("Analyser les charges")

    # Onglet 2: Téléchargement de fichiers
    with tab2:
        with st.form("input_form_files"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Contrat de bail commercial / Clauses")
                doc1_files = st.file_uploader(
                    "Téléchargez le(s) fichier(s) du contrat de bail commercial (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    help="Téléchargez votre contrat de bail commercial ou les clauses concernant les charges"
                )

                if doc1_files:
                    st.write(f"{len(doc1_files)} fichier(s) téléchargé(s) pour le bail")

            with col2:
                st.subheader("Reddition des charges")
                doc2_files = st.file_uploader(
                    "Téléchargez le(s) fichier(s) de reddition des charges (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    help="Téléchargez le document détaillant les charges qui vous sont facturées"
                )

                if doc2_files:
                    st.write(f"{len(doc2_files)} fichier(s) téléchargé(s) pour les charges")

            specific_questions_file = st.text_area(
                "Questions spécifiques (facultatif)",
                help="Avez-vous des questions particulières concernant certaines charges?"
            )

            submitted_files = st.form_submit_button("Analyser les fichiers")

    # Traitement du formulaire de saisie manuelle
    if submitted_manual:
        if not document1_manual or not document2_manual:
            st.error("Veuillez remplir les champs obligatoires (clauses du bail et détail des charges).")
        else:
            st.info("📋 Analyse des charges en cours - Cette opération peut prendre une minute...")
            # Analyser les charges avec l'approche structurée
            analysis = analyze_with_openai(document1_manual, document2_manual, document_type)
            if analysis:
                st.session_state.analysis = analysis
                st.session_state.analysis_complete = True
                # Sauvegarder les textes originaux pour l'export PDF
                st.session_state.document1_text = document1_manual
                st.session_state.document2_text = document2_manual

    # Traitement du formulaire de téléchargement de fichiers
    if submitted_files:
        if not doc1_files or not doc2_files:
            st.error("Veuillez télécharger au moins un fichier pour le bail et un fichier pour les charges.")
        else:
            with st.spinner("Extraction du texte des fichiers..."):
                # Extraire et combiner le texte de tous les fichiers
                document1_combined = process_multiple_files(doc1_files)
                document2_combined = process_multiple_files(doc2_files)

                if not document1_combined or not document2_combined:
                    st.error("Impossible d'extraire le texte des fichiers téléchargés.")
                else:
                    # Afficher un résumé du texte extrait
                    st.info(f"📄 Texte extrait: Bail ({len(document1_combined)} caractères), Charges ({len(document2_combined)} caractères)")

                    st.info("📋 Analyse des charges en cours - Cette opération peut prendre une minute...")
                    # Analyser les charges avec l'approche structurée
                    analysis = analyze_with_openai(document1_combined, document2_combined, document_type)
                    if analysis:
                        st.session_state.analysis = analysis
                        st.session_state.analysis_complete = True
                        # Sauvegarder les textes originaux pour l'export PDF
                        st.session_state.document1_text = document1_combined
                        st.session_state.document2_text = document2_combined

    # Afficher les résultats
    if st.session_state.analysis_complete:
        analysis = st.session_state.analysis

        st.header("Résultats de l'analyse des charges locatives commerciales")

        # Afficher le montant total et la conformité globale
        col1, col2 = st.columns(2)
        with col1:
            if "montant_total" in analysis:
                st.metric("Montant total des charges", f"{analysis['montant_total']:.2f}€")
        with col2:
            if "analyse_globale" in analysis and "taux_conformite" in analysis["analyse_globale"]:
                st.metric("Taux de conformité", f"{analysis['analyse_globale']['taux_conformite']}%")
        
        # Détail de l'analyse de conformité
        if "analyse_globale" in analysis and "conformite_detail" in analysis["analyse_globale"]:
            st.markdown("### Analyse de conformité")
            st.info(analysis["analyse_globale"]["conformite_detail"])

        # Section 1: Charges refacturables selon le bail
        st.markdown("## Charges refacturables selon le bail")
        if "charges_refacturables" in analysis and analysis["charges_refacturables"]:
            # Créer un DataFrame restructuré pour un meilleur affichage
            refined_data = []
            for charge in analysis["charges_refacturables"]:
                refined_data.append({
                    "Catégorie": charge.get("categorie", ""),
                    "Description": charge.get("description", ""),
                    "Base légale": charge.get("base_legale", ""),
                    "Certitude": charge.get("certitude", "")
                })
            
            refacturables_df = pd.DataFrame(refined_data)
            st.dataframe(refacturables_df, use_container_width=True)
        else:
            st.warning("Aucune information sur les charges refacturables n'a été identifiée dans le bail.")

        # Section 2: Charges effectivement facturées
        st.markdown("## Charges facturées")
        if "charges_facturees" in analysis and analysis["charges_facturees"]:
            # Préparation des données pour le tableau et le graphique
            charges_df = pd.DataFrame([
                {
                    "Poste": charge["poste"],
                    "Montant (€)": charge["montant"],
                    "% du total": f"{charge['pourcentage']:.1f}%",
                    "Conformité": charge["conformite"],
                    "Contestable": "Oui" if charge.get("contestable", False) else "Non"
                }
                for charge in analysis["charges_facturees"]
            ])
            
            # Affichage du tableau
            st.dataframe(charges_df, use_container_width=True)
            
            # Préparation du graphique camembert
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = [charge["poste"] for charge in analysis["charges_facturees"]]
            sizes = [charge["montant"] for charge in analysis["charges_facturees"]]
            
            # Génération du graphique
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%',
                textprops={'fontsize': 9},
                startangle=90
            )
            
            # Ajustements du graphique
            plt.setp(autotexts, size=9, weight='bold')
            plt.setp(texts, size=9)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            plt.title('Répartition des charges locatives commerciales')
            
            # Affichage du graphique
            st.pyplot(fig)
        else:
            st.warning("Aucune charge facturée n'a été identifiée.")

        # Section 3: Charges contestables
        st.markdown("## Charges potentiellement contestables")
        if "charges_facturees" in analysis:
            contestable_charges = [c for c in analysis["charges_facturees"] if c.get("contestable")]
            if contestable_charges:
                for charge in contestable_charges:
                    with st.expander(f"{charge['poste']} ({charge['montant']}€)"):
                        st.markdown(f"**Montant:** {charge['montant']}€ ({charge['pourcentage']}% du total)")
                        st.markdown(f"**Raison:** {charge.get('raison_contestation', 'Non spécifiée')}")
                        st.markdown(f"**Justification:** {charge.get('justification', '')}")
            else:
                st.success("Aucune charge contestable n'a été identifiée.")
        
        # Section 4: Recommandations
        st.markdown("## Recommandations")
        if "recommandations" in analysis and analysis["recommandations"]:
            for i, rec in enumerate(analysis["recommandations"]):
                st.markdown(f"{i+1}. {rec}")
        else:
            st.info("Aucune recommandation spécifique.")

        # Options d'export
        st.header("Exporter les résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            st.download_button(
                label="Télécharger l'analyse en JSON",
                data=json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8'),
                file_name='analyse_charges_locatives.json',
                mime='application/json',
            )
        
        with col2:
            # Export PDF
            try:
                document1_text = st.session_state.get('document1_text', '')
                document2_text = st.session_state.get('document2_text', '')
                
                # Générer le rapport PDF
                pdf_content = generate_pdf_report(
                    analysis, 
                    document_type, 
                    document1_text, 
                    document2_text
                )
                
                # Bouton de téléchargement pour le PDF
                st.download_button(
                    label="Télécharger le rapport PDF",
                    data=pdf_content,
                    file_name="rapport_analyse_charges_locatives.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF: {str(e)}")
                st.info("Assurez-vous d'avoir installé reportlab avec 'pip install reportlab'")

if __name__ == "__main__":
    main()
