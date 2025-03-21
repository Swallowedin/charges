import re
import json
import difflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import time

# ----------------- MODÈLES ET CONSTANTES -----------------

# Base de connaissances des charges locatives
@dataclass
class ChargeCategory:
    name: str                  # Nom standardisé
    keywords: List[str]        # Mots-clés associés
    description: str           # Description de la catégorie
    recoverable: bool = True   # Est-ce généralement récupérable ?
    legal_references: List[str] = None  # Références légales

# Définition des constantes
CHARGES_TYPES = {
    "commercial": [
        "Entretien et nettoyage des parties communes",
        "Eau et électricité des parties communes",
        "Ascenseurs et équipements techniques",
        "Espaces verts",
        "Sécurité et surveillance",
        "Gestion et honoraires",
        "Impôts et taxes",
        "Assurances"
    ],
    "habitation": [
        "Entretien des parties communes",
        "Eau",
        "Chauffage collectif",
        "Ascenseur",
        "Espaces verts",
        "Gardiennage"
    ]
}

CHARGES_CONTESTABLES = [
    "Grosses réparations (article 606 du Code civil)",
    "Remplacement d'équipements obsolètes",
    "Honoraires de gestion excessifs (>10% du montant des charges)",
    "Frais de personnel sans rapport avec l'immeuble",
    "Travaux d'amélioration (vs. entretien)",
    "Taxes normalement à la charge du propriétaire",
    "Assurance des murs et structure du bâtiment"
]

RATIOS_REFERENCE = {
    "commercial": {
        "charges/m2/an": {
            "min": 30,
            "max": 150,
            "median": 80
        },
        "honoraires gestion (% charges)": {
            "min": 2,
            "max": 8,
            "median": 5
        }
    },
    "habitation": {
        "charges/m2/an": {
            "min": 15,
            "max": 60,
            "median": 35
        }
    }
}

# Catégories standard de charges pour bail commercial
COMMERCIAL_CHARGES = [
    ChargeCategory(
        name="NETTOYAGE EXTERIEUR",
        keywords=["nettoyage", "propreté", "entretien extérieur", "nettoyage commun"],
        description="Nettoyage et entretien des parties communes extérieures",
    ),
    ChargeCategory(
        name="DECHETS SECS",
        keywords=["déchets", "ordures", "poubelles", "compacteurs", "traitement déchets"],
        description="Collecte et traitement des déchets",
    ),
    ChargeCategory(
        name="HYGIENE SANTE",
        keywords=["hygiène", "sanitaire", "désinfection", "dératisation", "assainissement"],
        description="Services liés à l'hygiène et à la santé",
    ),
    ChargeCategory(
        name="ELECTRICITE ABORDS & PKGS",
        keywords=["électricité", "éclairage", "énergie", "électrique", "alimentation"],
        description="Électricité des parties communes et parkings",
    ),
    ChargeCategory(
        name="STRUCTURE",
        keywords=["structure", "toiture", "façade", "gros œuvre", "fondations"],
        description="Entretien de la structure du bâtiment",
        recoverable=False,  # Généralement non récupérable (Art. 606)
        legal_references=["Art. 606 Code Civil"]
    ),
    ChargeCategory(
        name="VRD/ PKG/ SIGNAL.EXTERIEURE",
        keywords=["voirie", "signalisation", "parking", "circulation", "VRD", "réseaux"],
        description="Voirie, réseaux, parkings et signalisation extérieure",
    ),
    ChargeCategory(
        name="ESPACES VERTS EXTERIEURS",
        keywords=["espaces verts", "jardinage", "arbres", "pelouse", "paysagisme"],
        description="Entretien des espaces verts",
    ),
    ChargeCategory(
        name="MOYENS DE PROTECTION",
        keywords=["protection", "sécurité", "incendie", "alarme", "extincteurs"],
        description="Équipements et services de protection",
    ),
    ChargeCategory(
        name="SURVEILLANCE ABORDS/PKGS",
        keywords=["surveillance", "gardiennage", "vidéosurveillance", "sécurité", "gardien"],
        description="Surveillance des abords et parkings",
    ),
    ChargeCategory(
        name="GESTION ADMINISTRATION CENTRE",
        keywords=["administration", "gestion", "frais administratifs", "secrétariat"],
        description="Frais de gestion et d'administration",
    ),
    ChargeCategory(
        name="HONORAIRES GESTION",
        keywords=["honoraires", "frais de gestion", "property management", "syndic"],
        description="Honoraires de gestion de l'immeuble",
    ),
    ChargeCategory(
        name="TRAVAUX EXTERIEURS",
        keywords=["travaux", "réparations", "maintenance", "rénovation", "entretien"],
        description="Travaux d'entretien et de maintenance extérieurs",
    ),
    ChargeCategory(
        name="ASCENSEURS",
        keywords=["ascenseur", "élévateur", "monte-charge", "lift"],
        description="Entretien et maintenance des ascenseurs",
    ),
    ChargeCategory(
        name="ASSURANCES",
        keywords=["assurance", "prime", "garantie", "couverture", "police d'assurance"],
        description="Assurances liées à l'immeuble",
    ),
    ChargeCategory(
        name="CHAUFFAGE",
        keywords=["chauffage", "climatisation", "ventilation", "hvac", "chaleur"],
        description="Chauffage et climatisation",
    ),
    ChargeCategory(
        name="EAU",
        keywords=["eau", "distribution d'eau", "consommation eau", "plomberie"],
        description="Consommation et entretien des réseaux d'eau",
    ),
    ChargeCategory(
        name="TAXES",
        keywords=["taxe", "impôt", "contribution", "fiscalité", "redevance"],
        description="Taxes et impôts divers",
    ),
    ChargeCategory(
        name="SERVICES DIVERS",
        keywords=["divers", "autres services", "prestations", "fournitures"],
        description="Services divers non classifiés ailleurs",
    ),
]

# Critères de contestation des charges
CONTESTATION_CRITERIA = [
    {
        "name": "Grosses réparations (Art. 606)",
        "keywords": ["remplacement", "rénovation complète", "reconstruction", "article 606", "gros œuvre"],
        "description": "Grosses réparations relevant de l'article 606 du Code Civil"
    },
    {
        "name": "Travaux d'amélioration",
        "keywords": ["amélioration", "mise à niveau", "modernisation", "embellissement"],
        "description": "Travaux qui améliorent le bien au-delà de son état initial"
    },
    {
        "name": "Honoraires excessifs",
        "keywords": ["honoraires", "frais de gestion", "commission", "management fees"],
        "threshold": 10,  # Seuil de 10% du montant total des charges
        "description": "Honoraires dépassant 10% du montant total des charges"
    },
    {
        "name": "Charges propriétaire",
        "keywords": ["assurance murs", "impôt foncier", "ravalement", "structure", "toiture"],
        "description": "Charges incombant normalement au propriétaire"
    }
]

# ----------------- FONCTIONS D'EXTRACTION -----------------

# Extraction des charges avec regex
def extract_charges_fallback(text):
    """Extrait les charges du texte de la reddition avec regex"""
    charges = []
    lines = text.split('\n')
    current_category = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ligne contenant un montant en euros
        if '€' in line:
            # Essayer d'extraire le montant
            amount_match = re.search(r'(\d[\d\s]*[\d,\.]+)\s*€', line)
            if amount_match:
                amount_str = amount_match.group(1).replace(' ', '').replace(',', '.')
                try:
                    amount = float(amount_str)
                    # Extraire la description (tout ce qui précède le montant)
                    description = line[:amount_match.start()].strip()
                    if not description and current_category:
                        description = current_category
                    
                    charges.append({
                        "category": current_category,
                        "description": description,
                        "amount": amount
                    })
                except ValueError:
                    pass
        else:
            # Probablement une catégorie
            if ':' in line:
                current_category = line.split(':')[0].strip()
            elif line.isupper() or (len(line) > 3 and not any(c.isdigit() for c in line)):
                current_category = line
    
    return charges

def extract_relevant_sections(bail_text: str) -> str:
    """
    Extrait les sections pertinentes d'un bail en se concentrant sur les clauses 
    liées aux charges récupérables avec une approche adaptative.
    """
    # Mots-clés principaux pour identifier les sections relatives aux charges récupérables
    primary_keywords = [
        "charges récupérables", "charges locatives", "charges refacturables",
        "répartition des charges", "refacturation des charges"
    ]
    
    # Mots-clés secondaires qui, combinés avec d'autres indices, suggèrent une section pertinente
    secondary_keywords = [
        "charges", "dépenses", "frais", "entretien", "provision", "régularisation"
    ]
    
    # Contextes négatifs à exclure
    negative_contexts = [
        "à la charge exclusive du bailleur",
        "supporté entièrement par le bailleur",
        "à la charge du propriétaire",
        "charges sociales",
        "prendre en charge la TVA"
    ]
    
    # Patterns pour identifier les titres de section liés aux charges
    title_patterns = [
        r"(?i)article\s+\d+[\s.:-]*.*charges.*",
        r"(?i)chapitre\s+\d+[\s.:-]*.*charges.*",
        r"(?i)section\s+\d+[\s.:-]*.*charges.*",
        r"(?i)§\s*\d+[\s.:-]*.*charges.*",
        r"(?i)\d+[\s.:-]+.*charges.*",
        r"(?i)(charges|frais|dépenses)\s+locatives",
        r"(?i)répartition\s+des\s+(charges|frais|dépenses)",
        r"(?i)(entretien|maintenance)\s+et\s+réparations"
    ]
    
    # Diviser le texte en paragraphes
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', bail_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Premier passage : identifier les paragraphes hautement pertinents
    highly_relevant_indices = []
    for i, paragraph in enumerate(paragraphs):
        lower_paragraph = paragraph.lower()
        
        # Vérifier les mots-clés primaires
        if any(keyword.lower() in lower_paragraph for keyword in primary_keywords):
            highly_relevant_indices.append(i)
            continue
            
        # Vérifier les patterns de titre
        if any(re.search(pattern, paragraph) for pattern in title_patterns):
            highly_relevant_indices.append(i)
            continue
            
        # Vérifier la haute densité de mots-clés secondaires
        count_secondary = sum(1 for keyword in secondary_keywords if keyword.lower() in lower_paragraph)
        if count_secondary >= 2 and not any(neg.lower() in lower_paragraph for neg in negative_contexts):
            highly_relevant_indices.append(i)
    
    # Deuxième passage : ajouter le contexte aux paragraphes hautement pertinents
    relevant_indices = set(highly_relevant_indices)
    
    # Ajouter les paragraphes adjacents pour le contexte
    for i in highly_relevant_indices:
        # Ajouter le paragraphe précédent pour le contexte
        if i > 0:
            relevant_indices.add(i-1)
        # Ajouter les 2-3 paragraphes suivants qui pourraient contenir des détails
        for j in range(1, 4):
            if i+j < len(paragraphs):
                relevant_indices.add(i+j)
    
    # Reconstruire le texte en respectant l'ordre des paragraphes
    relevant_paragraphs = [paragraphs[i] for i in sorted(relevant_indices)]
    
    # Regarder le texte autour des occurrences de "article 606" ou "non récupérables"
    article_606_indices = []
    for i, paragraph in enumerate(paragraphs):
        if "article 606" in paragraph.lower() or "non récupérable" in paragraph.lower():
            article_606_indices.append(i)
            
    # Ajouter les paragraphes autour des mentions de l'article 606
    for i in article_606_indices:
        relevant_indices.update(range(max(0, i-1), min(i+3, len(paragraphs))))
    
    # Reconstruire le texte final
    final_relevant_paragraphs = [paragraphs[i] for i in sorted(relevant_indices)]
    extracted_text = "\n\n".join(final_relevant_paragraphs)
    
    # Si nous n'avons pas extrait grand-chose, élargir les critères
    if len(extracted_text) < 0.05 * len(bail_text):
        broader_indices = set()
        for i, paragraph in enumerate(paragraphs):
            if "charge" in paragraph.lower() or "dépense" in paragraph.lower() or "entretien" in paragraph.lower():
                broader_indices.add(i)
                if i > 0:
                    broader_indices.add(i-1)
                if i < len(paragraphs) - 1:
                    broader_indices.add(i+1)
        
        # Ajouter ces paragraphes à notre sélection
        final_relevant_paragraphs = list(set(final_relevant_paragraphs).union(
            [paragraphs[i] for i in broader_indices]
        ))
        # Trier pour maintenir l'ordre du document
        final_indices = [paragraphs.index(p) for p in final_relevant_paragraphs]
        final_relevant_paragraphs = [paragraphs[i] for i in sorted(final_indices)]
        extracted_text = "\n\n".join(final_relevant_paragraphs)
    
    return extracted_text

def extract_charges_from_document(text: str) -> List[Dict[str, Any]]:
    """
    Extrait les charges d'un document de reddition avec une approche adaptative 
    qui s'ajuste à différents formats.
    """
    charges = []
    
    # Approche 1: Format tabulaire structuré (comme dans les exemples)
    # Recherche des lignes avec description et montant
    pattern1 = r'(?P<category>[A-ZÀ-Ÿ\s/&.]+)\s+(?P<amount>[\d\s]+[,.]\d{2})\s*€?'
    matches1 = re.finditer(pattern1, text)
    
    for match in matches1:
        category = match.group('category').strip()
        amount_str = match.group('amount').replace(' ', '').replace(',', '.')
        
        try:
            amount = float(amount_str)
            # Éviter les doublons en vérifiant si cette catégorie existe déjà
            if not any(c['description'].lower() == category.lower() for c in charges):
                charges.append({
                    "category": find_best_category_match(category),
                    "description": category,
                    "amount": amount
                })
        except ValueError:
            continue
    
    # Approche 2: Format en liste ou avec préfixes (puces, tirets, etc.)
    pattern2 = r'(?:[-•*]|\d+[.)])\s*(?P<description>[^:]+)[:]\s*(?P<amount>[\d\s]+[,.]\d{2})\s*(?:€|EUR)'
    matches2 = re.finditer(pattern2, text)
    
    for match in matches2:
        description = match.group('description').strip()
        amount_str = match.group('amount').replace(' ', '').replace(',', '.')
        
        try:
            amount = float(amount_str)
            if not any(c['description'].lower() == description.lower() for c in charges):
                charges.append({
                    "category": find_best_category_match(description),
                    "description": description,
                    "amount": amount
                })
        except ValueError:
            continue
    
    # Approche 3: Recherche par paragraphe avec texte descriptif et montant
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
    for paragraph in paragraphs:
        # Rechercher un montant dans le paragraphe
        amount_match = re.search(r'(?:montant|somme|coût|total)?\s*(?:de|:)?\s*(?P<amount>[\d\s]+[,.]\d{2})\s*(?:€|EUR)', paragraph, re.IGNORECASE)
        if amount_match:
            amount_str = amount_match.group('amount').replace(' ', '').replace(',', '.')
            try:
                amount = float(amount_str)
                # Extraire une description du paragraphe
                desc_match = re.search(r'^(?P<desc>[^.]+)', paragraph.strip())
                if desc_match:
                    description = desc_match.group('desc').strip()
                    if not any(c['description'].lower() == description.lower() for c in charges):
                        charges.append({
                            "category": find_best_category_match(description),
                            "description": description,
                            "amount": amount
                        })
            except ValueError:
                continue
    
    # Approche 4: Lignes simples avec description et montant (format courant)
    pattern4 = r'(?P<description>[^:]+):\s*(?P<amount>[\d\s]+[,.]\d{2})\s*(?:€|EUR)?'
    matches4 = re.finditer(pattern4, text)
    
    for match in matches4:
        description = match.group('description').strip()
        amount_str = match.group('amount').replace(' ', '').replace(',', '.')
        
        try:
            amount = float(amount_str)
            if not any(c['description'].lower() == description.lower() for c in charges):
                charges.append({
                    "category": find_best_category_match(description),
                    "description": description,
                    "amount": amount
                })
        except ValueError:
            continue
    
    # Si le document contient un format qui ressemble à la reddition exemple
    pattern5 = r'([A-Z][A-Z\s/&]+)(?:\s*:\s*)?([\d\s]+[,.]\d{2})\s*€'
    matches5 = re.finditer(pattern5, text)
    
    for match in matches5:
        description = match.group(1).strip()
        amount_str = match.group(2).replace(' ', '').replace(',', '.')
        
        try:
            amount = float(amount_str)
            if not any(c['description'].lower() == description.lower() for c in charges):
                charges.append({
                    "category": find_best_category_match(description),
                    "description": description,
                    "amount": amount
                })
        except ValueError:
            continue
    
    # Si nous n'avons pas trouvé de charges, chercher des lignes qui semblent être des charges
    if not charges:
        # Chercher des lignes qui contiennent des mots clés de charges et des montants
        charge_keywords = [cat.name for cat in COMMERCIAL_CHARGES] + sum([cat.keywords for cat in COMMERCIAL_CHARGES], [])
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in charge_keywords):
                amount_match = re.search(r'(?P<amount>[\d\s]+[,.]\d{2})\s*(?:€|EUR)?', line)
                if amount_match:
                    amount_str = amount_match.group('amount').replace(' ', '').replace(',', '.')
                    try:
                        amount = float(amount_str)
                        description = re.sub(r'(?P<amount>[\d\s]+[,.]\d{2})\s*(?:€|EUR)?', '', line).strip()
                        if not any(c['description'].lower() == description.lower() for c in charges):
                            charges.append({
                                "category": find_best_category_match(description),
                                "description": description,
                                "amount": amount
                            })
                    except ValueError:
                        continue
    
    return charges

def find_best_category_match(description: str) -> str:
    """
    Trouve la meilleure catégorie standard correspondant à une description de charge.
    """
    description = description.lower()
    
    # D'abord, chercher une correspondance exacte dans les noms de catégorie
    for category in COMMERCIAL_CHARGES:
        if category.name.lower() == description:
            return category.name
    
    # Ensuite, vérifier les mots-clés
    for category in COMMERCIAL_CHARGES:
        for keyword in category.keywords:
            if keyword.lower() in description:
                return category.name
    
    # Si aucune correspondance directe, utiliser la similitude de texte
    best_match = None
    highest_score = 0
    
    for category in COMMERCIAL_CHARGES:
        # Calculer la similitude avec le nom de la catégorie
        name_score = difflib.SequenceMatcher(None, description, category.name.lower()).ratio()
        
        # Calculer la similitude avec chaque mot-clé
        keyword_scores = [difflib.SequenceMatcher(None, description, kw.lower()).ratio() 
                          for kw in category.keywords]
        max_keyword_score = max(keyword_scores) if keyword_scores else 0
        
        # Prendre le score le plus élevé entre le nom et les mots-clés
        score = max(name_score, max_keyword_score)
        
        if score > highest_score:
            highest_score = score
            best_match = category.name
    
    # Si le score est trop bas, classer comme "SERVICES DIVERS"
    if highest_score < 0.4:
        return "SERVICES DIVERS"
    
    return best_match

def analyze_charges_conformity(charges: List[Dict[str, Any]], bail_clauses: str) -> List[Dict[str, Any]]:
    """
    Analyse la conformité des charges avec les clauses du bail.
    """
    analyzed_charges = []
    total_amount = sum(charge["amount"] for charge in charges)
    
    for charge in charges:
        category = charge["category"]
        description = charge["description"]
        amount = charge["amount"]
        percentage = (amount / total_amount * 100) if total_amount > 0 else 0
        
        # Vérifier si la catégorie est mentionnée dans le bail
        category_in_bail = category.lower() in bail_clauses.lower() or any(
            keyword.lower() in bail_clauses.lower() 
            for cat in COMMERCIAL_CHARGES if cat.name == category
            for keyword in cat.keywords
        )
        
        # Vérifier la conformité avec les critères standards
        conformity_status = "conforme" if category_in_bail else "à vérifier"
        conformity_details = "Catégorie mentionnée dans le bail" if category_in_bail else "Catégorie non explicitement mentionnée dans le bail"
        
        # Chercher des clauses correspondantes dans le bail
        matching_clause = None
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', bail_clauses)
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            # Vérifier si le paragraphe contient la catégorie ou un mot-clé associé
            if category.lower() in paragraph_lower or any(
                keyword.lower() in paragraph_lower 
                for cat in COMMERCIAL_CHARGES if cat.name == category
                for keyword in cat.keywords
            ):
                matching_clause = paragraph.strip()
                break
        
        # Vérifier si la charge est potentiellement contestable
        contestable = False
        contestable_reason = None
        
        # Vérifier les critères de contestation
        for criterion in CONTESTATION_CRITERIA:
            criterion_match = False
            
            # Vérifier les mots-clés
            if any(kw.lower() in description.lower() for kw in criterion["keywords"]):
                criterion_match = True
            
            # Vérifier le seuil pour les honoraires
            if "threshold" in criterion and category == "HONORAIRES GESTION":
                if percentage > criterion["threshold"]:
                    criterion_match = True
            
            # Pour les grosses réparations (Article 606)
            if criterion["name"] == "Grosses réparations (Art. 606)":
                if "article 606" in bail_clauses.lower() and any(kw.lower() in description.lower() for kw in criterion["keywords"]):
                    criterion_match = True
                
                # Vérifier également la catégorie STRUCTURE qui est généralement non récupérable
                if category == "STRUCTURE":
                    for cat in COMMERCIAL_CHARGES:
                        if cat.name == "STRUCTURE" and not cat.recoverable:
                            criterion_match = True
            
            if criterion_match:
                contestable = True
                contestable_reason = criterion["description"]
                break
        
        analyzed_charge = {
            "category": category,
            "description": description,
            "amount": round(amount, 2),
            "percentage": round(percentage, 1),
            "conformity": conformity_status,
            "conformity_details": conformity_details,
            "contestable": contestable
        }
        
        if matching_clause:
            analyzed_charge["matching_clause"] = matching_clause
            
        if contestable and contestable_reason:
            analyzed_charge["contestable_reason"] = contestable_reason
        
        analyzed_charges.append(analyzed_charge)
    
    return analyzed_charges

def generate_global_analysis(analyzed_charges: List[Dict[str, Any]], bail_type: str, surface: Optional[float] = None) -> Dict[str, Any]:
    """
    Génère une analyse globale des charges.
    """
    total_amount = sum(charge["amount"] for charge in analyzed_charges)
    conforming_charges = [c for c in analyzed_charges if c["conformity"] == "conforme"]
    conformity_rate = (len(conforming_charges) / len(analyzed_charges) * 100) if analyzed_charges else 0
    
    contestable_charges = [c for c in analyzed_charges if c.get("contestable", False)]
    contestable_amount = sum(c["amount"] for c in contestable_charges)
    contestable_percentage = (contestable_amount / total_amount * 100) if total_amount > 0 else 0
    
    global_analysis = {
        "total_amount": round(total_amount, 2),
        "conformity_rate": round(conformity_rate, 1),
        "contestable_amount": round(contestable_amount, 2),
        "contestable_percentage": round(contestable_percentage, 1)
    }
    
    # Calculer le ratio de charges par m² si la surface est fournie
    if surface:
        try:
            surface_float = float(surface)
            charge_per_sqm = total_amount / surface_float
            global_analysis["charge_per_sqm"] = round(charge_per_sqm, 2)
            
            # Déterminer le réalisme des charges par rapport aux références du marché
            if bail_type == "commercial":
                min_ref = 30  # €/m²/an
                max_ref = 150  # €/m²/an
                median_ref = 80  # €/m²/an
            else:  # habitation
                min_ref = 15  # €/m²/an
                max_ref = 60  # €/m²/an
                median_ref = 35  # €/m²/an
            
            if charge_per_sqm < min_ref:
                realism = "bas"
                realism_details = f"Le montant des charges ({charge_per_sqm:.2f}€/m²/an) est inférieur à la référence minimale du marché ({min_ref}€/m²/an)"
            elif charge_per_sqm > max_ref:
                realism = "élevé"
                realism_details = f"Le montant des charges ({charge_per_sqm:.2f}€/m²/an) est supérieur à la référence maximale du marché ({max_ref}€/m²/an)"
            else:
                realism = "normal"
                realism_details = f"Le montant des charges ({charge_per_sqm:.2f}€/m²/an) est dans la fourchette normale du marché ({min_ref}-{max_ref}€/m²/an)"
            
            global_analysis["realism"] = realism
            global_analysis["realism_details"] = realism_details
        except ValueError:
            # Si la surface n'est pas un nombre valide
            pass
    
    return global_analysis

def generate_recommendations(analyzed_charges: List[Dict[str, Any]], global_analysis: Dict[str, Any], bail_clauses: str) -> List[str]:
    """
    Génère des recommandations basées sur l'analyse des charges.
    """
    recommendations = []
    
    # Recommandations basées sur le taux de conformité
    if global_analysis["conformity_rate"] < 50:
        recommendations.append("Demander au bailleur un détail précis des postes de charges et leur justification dans le bail.")
    
    # Recommandations basées sur les charges contestables
    contestable_charges = [c for c in analyzed_charges if c.get("contestable", False)]
    if contestable_charges:
        recommendations.append(f"Contester les charges suivantes qui représentent {global_analysis['contestable_percentage']:.1f}% du total: {', '.join([c['description'] for c in contestable_charges])}.")
    
    # Recommandations spécifiques par catégorie
    honoraires_gestion = [c for c in analyzed_charges if c["category"] == "HONORAIRES GESTION"]
    if honoraires_gestion:
        honoraires_percentage = sum(c["amount"] for c in honoraires_gestion) / global_analysis["total_amount"] * 100
        if honoraires_percentage > 10:
            recommendations.append(f"Les honoraires de gestion représentent {honoraires_percentage:.1f}% du total des charges, ce qui est supérieur au taux habituellement admis (8-10%). Demander une justification détaillée.")
    
    structure_charges = [c for c in analyzed_charges if c["category"] == "STRUCTURE"]
    if structure_charges:
        recommendations.append("Les charges liées à la structure du bâtiment relèvent généralement de l'article 606 du Code Civil et sont à la charge du propriétaire. Vérifier leur justification dans le bail.")
    
    # Vérifier si des documents justificatifs sont mentionnés dans le bail
    if "justificatif" in bail_clauses.lower() or "facture" in bail_clauses.lower():
        recommendations.append("Demander les justificatifs (factures, contrats) pour toutes les charges, comme prévu dans le bail.")
    else:
        recommendations.append("Demander systématiquement les justificatifs des charges, notamment pour les postes importants.")
    
    # Recommandation sur la procédure à suivre
    recommendations.append("Pour contester les charges non conformes, envoyer une lettre recommandée avec AR au bailleur en citant les clauses du bail et les dispositions légales applicables.")
    
    return recommendations

def analyze_charges_with_deterministic_approach(bail_clauses: str, charges_details: str, bail_type: str, surface: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyse les charges avec une approche déterministe pour garantir des résultats cohérents.
    """
    # 1. Extraire les sections pertinentes du bail
    relevant_bail_text = extract_relevant_sections(bail_clauses)
    
    # 2. Extraire les charges du document de reddition
    extracted_charges = extract_charges_from_document(charges_details)
    
    # 3. Analyser la conformité des charges
    analyzed_charges = analyze_charges_conformity(extracted_charges, relevant_bail_text)
    
    # 4. Générer l'analyse globale
    global_analysis = generate_global_analysis(analyzed_charges, bail_type, surface)
    
    # 5. Générer des recommandations
    recommendations = generate_recommendations(analyzed_charges, global_analysis, relevant_bail_text)
    
    # 6. Identifier les clauses pertinentes pour l'analyse
    clauses_analysis = extract_clauses_analysis(relevant_bail_text)
    
    # Construire le résultat final avec le format attendu par l'interface
    result = {
        "clauses_analysis": clauses_analysis,
        "charges_analysis": analyzed_charges,
        "global_analysis": global_analysis,
        "recommendations": recommendations,
        "extracted_sections": {
            "original_length": len(bail_clauses),
            "extracted_length": len(relevant_bail_text),
            "reduction_percent": round(100 - (len(relevant_bail_text) / len(bail_clauses) * 100), 1) if bail_clauses else 0
        }
    }
    
    return result

def extract_clauses_analysis(bail_text: str) -> List[Dict[str, str]]:
    """
    Extrait et structure les clauses du bail pour l'analyse.
    """
    clauses = []
    
    # Diviser en paragraphes significatifs
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', bail_text)
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Essayer d'identifier un titre
        title_match = re.search(r'(?i)^(article|chapitre|section|§)\s+\w+\s*[.:-]?\s*([^\n]+)', paragraph)
        
        if title_match:
            title = title_match.group(0).strip()
            text = paragraph[len(title_match.group(0)):].strip()
            
            clauses.append({
                "title": title,
                "text": text if text else paragraph
            })
        else:
            # Chercher d'autres formes de titres
            match = re.search(r'^([A-Z][^.]+)[.:]', paragraph)
            if match:
                title = match.group(1).strip()
                text = paragraph[len(match.group(0)):].strip()
                
                clauses.append({
                    "title": title,
                    "text": text if text else paragraph
                })
            else:
                # Pas de titre identifiable, utiliser les premiers mots
                words = paragraph.split()
                title = ' '.join(words[:min(5, len(words))]) + "..."
                
                clauses.append({
                    "title": title,
                    "text": paragraph
                })
    
    return clauses

def validate_and_normalize_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vérifie et normalise les résultats pour garantir la cohérence 
    et la reproductibilité de l'analyse.
    """
    if not result or "charges_analysis" not in result:
        return None
    
    charges = result["charges_analysis"]
    
    # Recalculer les montants totaux
    total_amount = sum(charge["amount"] for charge in charges)
    total_amount = round(total_amount, 2)  # Arrondir à 2 décimales pour cohérence
    
    # Normaliser et recalculer les pourcentages pour chaque charge
    for charge in charges:
        # Normaliser les montants à 2 décimales
        charge["amount"] = round(charge["amount"], 2)
        # Recalculer les pourcentages avec une précision fixe
        charge["percentage"] = round((charge["amount"] / total_amount * 100), 1) if total_amount > 0 else 0
    
    # Recalculer l'analyse globale
    contestable_charges = [c for c in charges if c.get("contestable", False)]
    contestable_amount = sum(c["amount"] for c in contestable_charges)
    contestable_amount = round(contestable_amount, 2)
    
    conforming_charges = [c for c in charges if c["conformity"] == "conforme"]
    conformity_rate = (len(conforming_charges) / len(charges) * 100) if charges else 0
    
    result["global_analysis"]["total_amount"] = total_amount
    result["global_analysis"]["contestable_amount"] = contestable_amount
    result["global_analysis"]["contestable_percentage"] = round((contestable_amount / total_amount * 100), 1) if total_amount > 0 else 0
    result["global_analysis"]["conformity_rate"] = round(conformity_rate, 1)
    
    # Vérifier que toutes les charges ont des catégories standard
    categories = [cat.name for cat in COMMERCIAL_CHARGES]
    for charge in charges:
        if charge["category"] not in categories:
            charge["category"] = "SERVICES DIVERS"  # Catégorie par défaut
    
    return result

def analyze_with_openai(bail_clauses, charges_details, bail_type, surface=None):
    """
    Version améliorée de l'analyse de bail qui utilise d'abord l'approche déterministe,
    puis fait appel à OpenAI uniquement si nécessaire ou sur demande spécifique.
    """
    try:
        # Informer l'utilisateur du début de l'analyse
        st.info("🔍 Début de l'analyse déterministe des charges locatives...")
        
        # Extraire les sections pertinentes du bail
        relevant_bail_text = extract_relevant_sections(bail_clauses)
        
        # Informer l'utilisateur de l'optimisation
        original_length = len(bail_clauses)
        extracted_length = len(relevant_bail_text)
        reduction_percent = round(100 - (extracted_length / original_length * 100), 1) if original_length > 0 else 0
        
        st.info(f"🔍 Optimisation du bail : {original_length:,} caractères → {extracted_length:,} caractères ({reduction_percent}% de réduction)")
        
        # Afficher un aperçu des sections extraites
        with st.expander("Aperçu des sections pertinentes extraites"):
            st.text(relevant_bail_text[:1000] + "..." if len(relevant_bail_text) > 1000 else relevant_bail_text)
        
        # Lancer l'analyse déterministe
        start_time = time.time()
        
        # ÉTAPE 1: Analyser avec l'approche déterministe
        deterministic_result = analyze_charges_with_deterministic_approach(bail_clauses, charges_details, bail_type, surface)
        deterministic_result = validate_and_normalize_results(deterministic_result)
        
        # Mesurer le temps d'exécution
        execution_time = time.time() - start_time
        st.success(f"✅ Analyse déterministe terminée en {execution_time:.2f} secondes")
        
        # Vérifier si les résultats déterministes sont suffisants
        if deterministic_result and len(deterministic_result["charges_analysis"]) > 0:
            return deterministic_result
        
        # ÉTAPE 2: Si l'analyse déterministe échoue ou identifie trop peu de charges, utiliser OpenAI
        st.warning("⚠️ L'analyse déterministe n'a pas identifié suffisamment de charges. Tentative avec GPT-4o-mini...")
        
        # Préparation du prompt OpenAI
        prompt = f"""
        # Analyse de charges locatives
        
        ## Contexte
        Bail {bail_type}, analyse des charges refacturées vs clauses du bail.
        IMPORTANT: Le texte du bail a été extrait pour se concentrer sur les clauses pertinentes liées aux charges.

        ## Référentiel
        Charges habituellement refacturables: {', '.join(CHARGES_TYPES[bail_type])}
        Charges contestables: {', '.join(CHARGES_CONTESTABLES)}

        ## Clauses du bail (sections pertinentes)
        {relevant_bail_text}

        ## Charges refacturées
        {charges_details}

        ## Surface: {surface if surface else "Non spécifiée"}

        ## Tâche
        1. Extraire clauses et charges avec montants PRÉCIS (2 décimales)
        2. Analyser conformité de chaque charge avec le bail
        3. Identifier charges contestables
        4. Calculer total et ratio/m2 si surface fournie
        5. Analyser réalisme: Commercial {RATIOS_REFERENCE['commercial']['charges/m2/an']['min']}-{RATIOS_REFERENCE['commercial']['charges/m2/an']['max']}€/m2/an, Habitation {RATIOS_REFERENCE['habitation']['charges/m2/an']['min']}-{RATIOS_REFERENCE['habitation']['charges/m2/an']['max']}€/m2/an
        6. Formuler recommandations

        ## Format JSON
        {{
            "clauses_analysis":[{{"title":"","text":""}}],
            "charges_analysis":[{{"category":"","description":"","amount":0.00,"percentage":0.0,"conformity":"conforme|à vérifier","conformity_details":"","matching_clause":"","contestable":true|false,"contestable_reason":""}}],
            "global_analysis":{{"total_amount":0.00,"charge_per_sqm":0.00,"conformity_rate":0.0,"realism":"normal|bas|élevé","realism_details":""}},
            "recommendations":[""]
        }}
        
        IMPORTANT: Sois EXTRÊMEMENT rigoureux et déterministe dans ton analyse pour garantir des résultats constants et reproductibles.
        """

        # Essayer d'abord avec gpt-4o-mini
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Température 0 pour maximiser la cohérence
                response_format={"type": "json_object"}  # Forcer une réponse JSON
            )
            result = json.loads(response.choices[0].message.content)
            st.success("✅ Analyse complémentaire réalisée avec gpt-4o-mini")
            
            # Fusionner les résultats si l'analyse déterministe avait trouvé quelque chose
            if deterministic_result and len(deterministic_result["charges_analysis"]) > 0:
                # Garder les charges identifiées par l'approche déterministe
                openai_charges = {c["description"].lower(): c for c in result["charges_analysis"]}
                for charge in deterministic_result["charges_analysis"]:
                    if charge["description"].lower() not in openai_charges:
                        result["charges_analysis"].append(charge)
                
                # Récalculer les pourcentages et totaux
                result = validate_and_normalize_results(result)
                
                st.info("🔄 Résultats de l'analyse déterministe et de l'analyse GPT fusionnés")
            
            return result
            
        except Exception as e:
            st.warning(f"⚠️ Erreur avec gpt-4o-mini: {str(e)}. Tentative avec gpt-3.5-turbo...")
            
            # Si échec, basculer vers gpt-3.5-turbo
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Température 0 pour maximiser la cohérence
                    response_format={"type": "json_object"}  # Forcer une réponse JSON
                )
                
                result = json.loads(response.choices[0].message.content)
                st.success("✅ Analyse complémentaire réalisée avec gpt-3.5-turbo")
                
                # Fusionner aussi ici si nécessaire
                if deterministic_result and len(deterministic_result["charges_analysis"]) > 0:
                    openai_charges = {c["description"].lower(): c for c in result["charges_analysis"]}
                    for charge in deterministic_result["charges_analysis"]:
                        if charge["description"].lower() not in openai_charges:
                            result["charges_analysis"].append(charge)
                    
                    result = validate_and_normalize_results(result)
                    st.info("🔄 Résultats de l'analyse déterministe et de l'analyse GPT fusionnés")
                
                return result
                
            except Exception as e2:
                st.error(f"❌ Erreur avec gpt-3.5-turbo: {str(e2)}. Utilisation des résultats de l'analyse déterministe uniquement.")
                
                # Si l'approche OpenAI échoue complètement, retourner les résultats déterministes
                if deterministic_result:
                    return deterministic_result
                else:
                    # Fallback absolu en cas d'échec total
                    return fallback_analysis(bail_clauses, charges_details, bail_type, surface)

    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        return fallback_analysis(bail_clauses, charges_details, bail_type, surface)

def fallback_analysis(bail_clauses, charges_details, bail_type, surface=None):
    """
    Analyse de secours simplifiée en cas d'échec des méthodes principales.
    """
    try:
        charges = extract_charges_fallback(charges_details)
        total_amount = sum(charge["amount"] for charge in charges)

        return {
            "clauses_analysis": [{"title": "Clause extraite manuellement", "text": clause.strip()} for clause in bail_clauses.split('\n') if clause.strip()],
            "charges_analysis": [
                {
                    "category": charge["category"] if charge["category"] else "SERVICES DIVERS",
                    "description": charge["description"],
                    "amount": charge["amount"],
                    "percentage": (charge["amount"] / total_amount * 100) if total_amount > 0 else 0,
                    "conformity": "à vérifier",
                    "conformity_details": "Analyse de secours (méthodes principales indisponibles)",
                    "matching_clause": None,
                    "contestable": False,
                    "contestable_reason": None
                } for charge in charges
            ],
            "global_analysis": {
                "total_amount": total_amount,
                "charge_per_sqm": total_amount / float(surface) if surface and surface.replace('.', '').isdigit() else None,
                "conformity_rate": 0,
                "realism": "indéterminé",
                "realism_details": "Analyse de secours (méthodes principales indisponibles)"
            },
            "recommendations": [
                "Vérifier manuellement la conformité des charges avec les clauses du bail",
                "Demander des justificatifs détaillés pour toutes les charges importantes"
            ]
        }
    except Exception as fallback_error:
        st.error(f"❌ Erreur lors de l'analyse de secours: {str(fallback_error)}")
        # Retour minimal en cas d'échec total
        return {
            "clauses_analysis": [],
            "charges_analysis": [],
            "global_analysis": {
                "total_amount": 0,
                "conformity_rate": 0,
                "realism": "indéterminé",
                "realism_details": "Analyse impossible"
            },
            "recommendations": [
                "L'analyse automatique a échoué. Veuillez vérifier le format de vos documents.",
                "Essayez de copier-coller directement le texte plutôt que d'utiliser des fichiers."
            ]
        }

# ---- MODIFICATION DE LA FONCTION MAIN POUR INTÉGRER L'APPROCHE DÉTERMINISTE ----

def main():
    st.title("Analyseur de Charges Locatives")
    st.markdown("""
    Cet outil analyse la cohérence entre les charges refacturées par votre bailleur 
    et les clauses de votre contrat de bail avec une approche déterministe renforcée par IA.
    """)

    # Sidebar pour la configuration
    st.sidebar.header("Configuration")

    bail_type = st.sidebar.selectbox(
        "Type de bail",
        options=["commercial", "habitation"],
        index=0
    )

    surface = st.sidebar.text_input(
        "Surface locative (m²)",
        help="Utilisé pour calculer le ratio de charges au m²"
    )
    
    # Option pour forcer l'utilisation de l'IA
    use_openai = st.sidebar.checkbox(
        "Utiliser l'IA pour l'analyse",
        value=True,
        help="Active l'analyse par GPT-4o-mini en complément de l'analyse déterministe"
    )

    # Interface principale avec onglets
    tab1, tab2 = st.tabs(["Saisie manuelle", "Téléchargement de fichiers"])

    # Onglet 1: Saisie manuelle
    with tab1:
        with st.form("input_form_manual"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Clauses du bail concernant les charges")
                bail_clauses_manual = st.text_area(
                    "Copiez-collez les clauses du bail concernant les charges refacturables",
                    height=250,
                    help="Utilisez un format avec une clause par ligne, commençant par •, - ou un numéro"
                )

            with col2:
                st.subheader("Détail des charges refacturées")
                charges_details_manual = st.text_area(
                    "Entrez le détail des charges (poste et montant)",
                    height=250,
                    help="Format recommandé: une charge par ligne avec le montant en euros (ex: 'Nettoyage: 1200€')"
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
                st.subheader("Documents du bail")
                bail_files = st.file_uploader(
                    "Téléchargez le(s) document(s) du bail (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    help="Téléchargez un ou plusieurs documents contenant les clauses du bail"
                )

                if bail_files:
                    st.write(f"{len(bail_files)} fichier(s) téléchargé(s) pour le bail")
                    with st.expander("Aperçu des fichiers du bail"):
                        for file in bail_files:
                            st.write(f"**{file.name}**")
                            display_file_preview(file)
                            st.markdown("---")

            with col2:
                st.subheader("Documents des charges")
                charges_files = st.file_uploader(
                    "Téléchargez le(s) document(s) des charges (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    help="Téléchargez un ou plusieurs documents contenant le détail des charges"
                )

                if charges_files:
                    st.write(f"{len(charges_files)} fichier(s) téléchargé(s) pour les charges")
                    with st.expander("Aperçu des fichiers des charges"):
                        for file in charges_files:
                            st.write(f"**{file.name}**")
                            display_file_preview(file)
                            st.markdown("---")

            specific_questions_file = st.text_area(
                "Questions spécifiques (facultatif)",
                help="Avez-vous des questions particulières concernant certaines charges?"
            )

            submitted_files = st.form_submit_button("Analyser les fichiers")

    # Traitement du formulaire de saisie manuelle
    if submitted_manual:
        if not bail_clauses_manual or not charges_details_manual:
            st.error("Veuillez remplir les champs obligatoires (clauses du bail et détail des charges).")
        else:
            with st.spinner("Analyse en cours..."):
                if not use_openai:
                    # Utiliser uniquement l'analyse déterministe
                    analysis = analyze_charges_with_deterministic_approach(bail_clauses_manual, charges_details_manual, bail_type, surface)
                    analysis = validate_and_normalize_results(analysis)
                    st.success("✅ Analyse déterministe terminée avec succès")
                else:
                    # Utiliser l'analyse combinée (déterministe + IA)
                    analysis = analyze_with_openai(bail_clauses_manual, charges_details_manual, bail_type, surface)
                
                if analysis:
                    st.session_state.analysis = analysis
                    st.session_state.analysis_complete = True
                    # Sauvegarder les textes originaux pour l'export PDF
                    st.session_state.bail_text = bail_clauses_manual
                    st.session_state.charges_text = charges_details_manual

    # Traitement du formulaire de téléchargement de fichiers
    if submitted_files:
        if not bail_files or not charges_files:
            st.error("Veuillez télécharger au moins un fichier pour le bail et un fichier pour les charges.")
        else:
            with st.spinner("Extraction et analyse des fichiers en cours..."):
                # Extraire et combiner le texte de tous les fichiers
                bail_clauses_combined = process_multiple_files(bail_files)
                charges_details_combined = process_multiple_files(charges_files)

                if not bail_clauses_combined or not charges_details_combined:
                    st.error("Impossible d'extraire le texte des fichiers téléchargés.")
                else:
                    # Afficher le texte extrait pour vérification
                    with st.expander("Texte extrait du bail"):
                        st.text(bail_clauses_combined[:2000] + "..." if len(bail_clauses_combined) > 2000 else bail_clauses_combined)

                    with st.expander("Texte extrait des charges"):
                        st.text(charges_details_combined[:2000] + "..." if len(charges_details_combined) > 2000 else charges_details_combined)

                    # Analyser les charges avec l'approche appropriée
                    if not use_openai:
                        # Utiliser uniquement l'analyse déterministe
                        analysis = analyze_charges_with_deterministic_approach(bail_clauses_combined, charges_details_combined, bail_type, surface)
                        analysis = validate_and_normalize_results(analysis)
                        st.success("✅ Analyse déterministe terminée avec succès")
                    else:
                        # Utiliser l'analyse combinée (déterministe + IA)
                        analysis = analyze_with_openai(bail_clauses_combined, charges_details_combined, bail_type, surface)
                    
                    if analysis:
                        st.session_state.analysis = analysis
                        st.session_state.analysis_complete = True
                        # Sauvegarder les textes originaux pour l'export PDF
                        st.session_state.bail_text = bail_clauses_combined
                        st.session_state.charges_text = charges_details_combined

    # Le reste du code de la fonction main() reste inchangé
    # Afficher les résultats
    if st.session_state.analysis_complete:
        display_analysis_results(st.session_state.analysis, bail_type, surface)

def display_analysis_results(analysis, bail_type, surface=None):
    """
    Affiche les résultats de l'analyse dans l'interface Streamlit.
    """
    charges_analysis = analysis["charges_analysis"]
    global_analysis = analysis["global_analysis"]

    st.header("Résultats de l'analyse")

    # Résumé global
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Montant total des charges", f"{global_analysis['total_amount']:.2f}€")
    with col2:
        if global_analysis.get('charge_per_sqm'):
            st.metric("Charges au m²/an", f"{global_analysis['charge_per_sqm']:.2f}€")
    with col3:
        st.metric("Taux de conformité", f"{global_analysis['conformity_rate']:.0f}%")

    # Alerte sur le réalisme
    if global_analysis.get('realism') != "indéterminé":
        color_map = {"normal": "success", "bas": "info", "élevé": "warning"}
        alert_type = color_map.get(global_analysis.get('realism'), "info")
        if alert_type == "success":
            st.success(global_analysis['realism_details'])
        elif alert_type == "info":
            st.info(global_analysis['realism_details'])
        else:
            st.warning(global_analysis['realism_details'])

    # Visualisation graphique
    st.subheader("Répartition des charges")
    fig = plot_charges_breakdown(charges_analysis)
    if fig:
        st.pyplot(fig)

    # Tableau d'analyse détaillée
    st.subheader("Analyse détaillée des charges")

    # Créer DataFrame pour affichage
    df = pd.DataFrame([
        {
            "Description": charge["description"],
            "Montant (€)": f"{charge['amount']:.2f}",
            "% du total": f"{charge['percentage']:.1f}%",
            "Conformité": charge["conformity"],
            "Détails": charge.get("conformity_details", ""),
            "Contestable": "Oui" if charge.get("contestable") else "Non"
        }
        for charge in charges_analysis
    ])

    # Afficher le DataFrame
    st.dataframe(df)

    # Charges contestables
    contestable_charges = [c for c in charges_analysis if c.get("contestable", False)]
    if contestable_charges:
        st.subheader("Charges potentiellement contestables")
        for i, charge in enumerate(contestable_charges):
            with st.expander(f"{charge['description']} ({charge['amount']:.2f}€)"):
                st.markdown(f"**Montant:** {charge['amount']:.2f}€ ({charge['percentage']:.1f}% du total)")

                if "contestable_reason" in charge and charge["contestable_reason"]:
                    st.markdown(f"**Raison:** {charge['contestable_reason']}")
                else:
                    st.markdown(f"**Raison:** {charge.get('conformity_details', 'Non spécifiée')}")

                if "matching_clause" in charge and charge["matching_clause"]:
                    st.markdown(f"""
                    **Clause correspondante dans le bail:**
                    >{charge['matching_clause']}
                    """)

    # Recommandations
    st.subheader("Recommandations")
    recommendations = analysis["recommendations"]
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")

    # Options d'export
    st.header("Exporter les résultats")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export JSON
        st.download_button(
            label="Télécharger l'analyse en JSON",
            data=json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8'),
            file_name='analyse_charges.json',
            mime='application/json',
        )
    
    with col2:
        # Export PDF
        try:
            bail_text = st.session_state.get('bail_text', '')
            charges_text = st.session_state.get('charges_text', '')
            
            # Générer le rapport PDF
            pdf_content = generate_pdf_report(
                analysis, 
                bail_type, 
                surface, 
                bail_text, 
                charges_text
            )
            
            # Bouton de téléchargement pour le PDF
            st.download_button(
                label="Télécharger le rapport PDF",
                data=pdf_content,
                file_name="rapport_charges_locatives.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {str(e)}")
            st.info("Assurez-vous d'avoir installé reportlab avec 'pip install reportlab'")

if __name__ == "__main__":
    main()
