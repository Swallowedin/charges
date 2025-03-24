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
    page_title="Analyseur de Charges Locatives avec GPT-4o-mini",
    page_icon="📊",
    layout="wide"
)

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

# Initialisation de l'état de la session
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Configuration de l'API OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("⚠️ La clé API OpenAI n'est pas définie. Veuillez la configurer dans les variables d'environnement ou les secrets.")
    st.stop()

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

def display_file_preview(uploaded_file):
    """Afficher un aperçu du fichier selon son type"""
    if uploaded_file is None:
        return
        
    file_type = uploaded_file.type
    
    if file_type.startswith("image/"):
        st.image(uploaded_file, caption=f"Aperçu: {uploaded_file.name}", use_column_width=True)
    elif file_type == "application/pdf":
        # Créer un lien pour visualiser le PDF
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.write(f"Aperçu non disponible pour {uploaded_file.name} (type: {file_type})")

def extract_relevant_sections(bail_text):
    """
    Extrait les sections pertinentes d'un bail volumineux en se concentrant sur les clauses liées aux charges.
    """
    # Mots-clés pour identifier les sections relatives aux charges
    keywords = [
        "charges", "charges locatives", "charges récupérables", 
        "dépenses communes", "charges communes", "répartition des charges",
        "provision", "régularisation", "article 606", "article 605",
        "frais", "honoraires", "dépenses", "refacturation", 
        "parties communes", "loyer et charges", "reddition", "décompte"
    ]
    
    # Identifier les titres potentiels des sections
    title_patterns = [
        r"(?i)article\s+\d+\s*[:-]?\s*(.*charges.*|.*récupéra.*|.*dépenses.*)",
        r"(?i)chapitre\s+\d+\s*[:-]?\s*(.*charges.*|.*récupéra.*|.*dépenses.*)",
        r"(?i)section\s+\d+\s*[:-]?\s*(.*charges.*|.*récupéra.*|.*dépenses.*)",
        r"(?i)(charges|frais|dépenses)\s+\w+",
        r"(?i)répartition\s+des\s+(charges|frais|dépenses)",
        r"(?i)(provision|régularisation)\s+\w+"
    ]
    
    # Diviser le texte en lignes
    lines = bail_text.split('\n')
    
    # Extraire les sections pertinentes
    relevant_sections = []
    current_section = []
    in_relevant_section = False
    
    for line in lines:
        line_lower = line.lower()
        
        # Déterminer si cette ligne marque le début d'une section pertinente
        is_start_of_relevant_section = False
        
        # Vérifier si la ligne contient un mot-clé
        if any(keyword in line_lower for keyword in keywords):
            is_start_of_relevant_section = True
        
        # Vérifier si la ligne correspond à un pattern de titre
        for pattern in title_patterns:
            if re.search(pattern, line):
                is_start_of_relevant_section = True
                break
        
        # Traiter la ligne selon son contexte
        if is_start_of_relevant_section:
            # Si on était déjà dans une section pertinente, sauvegarder la section précédente
            if in_relevant_section and current_section:
                relevant_sections.append('\n'.join(current_section))
                current_section = []
            
            # Commencer une nouvelle section
            in_relevant_section = True
            current_section.append(line)
            
            # Nombre de lignes à capturer après un mot-clé
            lines_to_capture = 20
            
        elif in_relevant_section:
            current_section.append(line)
            
            # Après avoir capturé assez de lignes, vérifier si on continue
            if len(current_section) > lines_to_capture:
                # Si on trouve un nouveau titre ou une ligne vide, terminer la section
                if re.match(r"(?i)^(article|chapitre|section)\s+\d+", line) or line.strip() == "":
                    in_relevant_section = False
                    relevant_sections.append('\n'.join(current_section))
                    current_section = []
    
    # Ne pas oublier la dernière section
    if in_relevant_section and current_section:
        relevant_sections.append('\n'.join(current_section))
    
    # Combiner toutes les sections pertinentes en un seul texte
    extracted_text = "\n\n".join(relevant_sections)
    
    return extracted_text

def analyze_with_gpt4o_mini(bail_clauses, charges_details, bail_type, surface=None):
    """
    Analyse optimisée avec GPT-4o-mini pour maximiser la cohérence et la précision.
    """
    try:
        # Extraire les sections pertinentes du bail pour réduire le bruit
        relevant_bail_text = extract_relevant_sections(bail_clauses)
        
        # Informer l'utilisateur de l'optimisation
        original_length = len(bail_clauses)
        extracted_length = len(relevant_bail_text)
        reduction_percent = round(100 - (extracted_length / original_length * 100), 1) if original_length > 0 else 0
        
        st.info(f"🔍 Optimisation du bail : {original_length:,} caractères → {extracted_length:,} caractères ({reduction_percent}% de réduction)")
        
        # Afficher un aperçu des sections extraites
        with st.expander("Aperçu des sections pertinentes extraites"):
            st.text(relevant_bail_text[:1000] + "..." if len(relevant_bail_text) > 1000 else relevant_bail_text)
        
        # Prompt optimisé pour GPT-4o-mini avec focus sur la précision et la cohérence
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
        1. ÉTAPE 1: Extraire avec GRANDE PRÉCISION (2 décimales) toutes les charges mentionnées avec leur montant exact
        2. ÉTAPE 2: Pour chaque charge, chercher minutieusement dans le bail s'il y a une clause qui l'autorise
        3. ÉTAPE 3: Vérifier si la charge est contestable selon les critères du référentiel fourni
        4. ÉTAPE 4: Calculer le montant total exact des charges et le ratio par m² si la surface est fournie
        5. ÉTAPE 5: Analyser si le montant est réaliste selon ces références:
           Commercial: {RATIOS_REFERENCE['commercial']['charges/m2/an']['min']}-{RATIOS_REFERENCE['commercial']['charges/m2/an']['max']}€/m²/an
           Habitation: {RATIOS_REFERENCE['habitation']['charges/m2/an']['min']}-{RATIOS_REFERENCE['habitation']['charges/m2/an']['max']}€/m²/an
        6. ÉTAPE 6: Formuler des recommandations pratiques et actionnables

        ## Consignes CRUCIALES pour la précision et cohérence
        1. TOUS les montants doivent avoir EXACTEMENT 2 décimales (xx.xx)
        2. TOUS les pourcentages doivent avoir EXACTEMENT 1 décimale (xx.x%)
        3. Les statuts de conformité sont BINAIRES: "conforme" OU "à vérifier" (pas d'autres valeurs)
        4. Les motifs de contestation doivent être SPÉCIFIQUES et référencés
        5. Le montant total doit être MATHÉMATIQUEMENT EXACT (somme précise des montants)
        6. Les analyses doivent être FACTUELLES et basées sur les clauses réelles du bail
        7. Soyez SYSTÉMATIQUE et MÉTHODIQUE dans l'analyse
        8. Attribue une catégorie standard à chaque charge (utilise "SERVICES DIVERS" si aucune ne correspond)
        9. Le calcul du pourcentage de chaque charge doit être EXACT et basé sur la formule: (montant_charge / montant_total) * 100

        ## Format JSON rigoureux
        {
            "clauses_analysis":[{"title":"","text":""}],
            "charges_analysis":[
                {
                    "category":"",
                    "description":"",
                    "amount":0.00,
                    "percentage":0.0,
                    "conformity":"conforme|à vérifier",
                    "conformity_details":"",
                    "matching_clause":"",
                    "contestable":true|false,
                    "contestable_reason":""
                }
            ],
            "global_analysis":{
                "total_amount":0.00,
                "charge_per_sqm":0.00,
                "conformity_rate":0.0,
                "contestable_amount":0.00,
                "contestable_percentage":0.0,
                "realism":"normal|bas|élevé",
                "realism_details":""
            },
            "recommendations":[""]
        }
        
        IMPORTANT: Une analyse CORRECTE, COMPLÈTE et COHÉRENTE est IMPÉRATIVE. Suivez STRICTEMENT les consignes ci-dessus.
        """

        # Appel à GPT-4o-mini avec température à 0 pour maximiser le déterminisme
        st.info("⏳ Analyse avec GPT-4o-mini en cours...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Température 0 pour maximiser la cohérence
            response_format={"type": "json_object"},  # Forcer une réponse JSON
            seed=42  # Utiliser une graine fixe pour renforcer la cohérence
        )
        
        # Traiter la réponse
        result = json.loads(response.choices[0].message.content)
        st.success("✅ Analyse terminée avec succès")
        
        # Vérifier et normaliser les résultats pour assurer la cohérence
        result = normalize_analysis_results(result)
        
        return result

    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        return generate_fallback_analysis(bail_clauses, charges_details, bail_type, surface)

def normalize_analysis_results(result):
    """
    Normalise les résultats pour assurer la cohérence et la précision absolue.
    """
    if not result or "charges_analysis" not in result:
        return result
    
    charges = result.get("charges_analysis", [])
    
    # Vérifier si les charges sont vides
    if not charges:
        return result
    
    # Recalculer les montants totaux avec précision
    total_amount = sum(charge.get("amount", 0) for charge in charges)
    total_amount = round(total_amount, 2)  # Arrondir à 2 décimales pour cohérence
    
    # Normaliser et recalculer les pourcentages pour chaque charge
    for charge in charges:
        # S'assurer que les montants sont des nombres et arrondis à 2 décimales
        charge["amount"] = round(float(charge.get("amount", 0)), 2)
        # Recalculer les pourcentages avec précision
        charge["percentage"] = round((charge["amount"] / total_amount * 100), 1) if total_amount > 0 else 0.0
        # S'assurer que conformity est soit "conforme" soit "à vérifier"
        if "conformity" in charge and charge["conformity"] not in ["conforme", "à vérifier"]:
            charge["conformity"] = "à vérifier"
    
    # Vérifier et recalculer les métriques globales
    if "global_analysis" in result:
        global_analysis = result["global_analysis"]
        global_analysis["total_amount"] = total_amount
        
        # Recalculer le taux de conformité
        conforming_charges = [c for c in charges if c.get("conformity") == "conforme"]
        conformity_rate = (len(conforming_charges) / len(charges) * 100) if charges else 0
        global_analysis["conformity_rate"] = round(conformity_rate, 1)
        
        # Recalculer les montants contestables
        contestable_charges = [c for c in charges if c.get("contestable", False)]
        contestable_amount = sum(c["amount"] for c in contestable_charges)
        global_analysis["contestable_amount"] = round(contestable_amount, 2)
        global_analysis["contestable_percentage"] = round((contestable_amount / total_amount * 100), 1) if total_amount > 0 else 0.0
        
        # S'assurer que les montants par mètre carré sont correctement calculés
        if "charge_per_sqm" in global_analysis and global_analysis["charge_per_sqm"]:
            global_analysis["charge_per_sqm"] = round(global_analysis["charge_per_sqm"], 2)
    
    # Vérifier que chaque charge a tous les champs requis
    for charge in charges:
        if "contestable" not in charge:
            charge["contestable"] = False
        if "contestable_reason" not in charge and charge["contestable"]:
            charge["contestable_reason"] = "Raison non spécifiée"
    
    return result

def generate_fallback_analysis(bail_clauses, charges_details, bail_type, surface=None):
    """
    Analyse de secours simplifiée en cas d'échec de l'analyse principale.
    """
    try:
        # Tenter d'extraire les charges avec un regex basique
        charges = extract_charges_fallback(charges_details)
        total_amount = sum(charge["amount"] for charge in charges)

        # Créer une structure minimale de résultat
        return {
            "clauses_analysis": [{"title": "Clause extraite", "text": clause.strip()} for clause in bail_clauses.split('\n\n')[:5] if clause.strip()],
            "charges_analysis": [
                {
                    "category": "SERVICES DIVERS",
                    "description": charge["description"],
                    "amount": round(charge["amount"], 2),
                    "percentage": round((charge["amount"] / total_amount * 100), 1) if total_amount > 0 else 0.0,
                    "conformity": "à vérifier",
                    "conformity_details": "Analyse de secours automatique",
                    "matching_clause": None,
                    "contestable": False,
                    "contestable_reason": None
                } for charge in charges
            ],
            "global_analysis": {
                "total_amount": round(total_amount, 2),
                "charge_per_sqm": round(total_amount / float(surface), 2) if surface and surface.replace('.', '').isdigit() else None,
                "conformity_rate": 0.0,
                "contestable_amount": 0.00,
                "contestable_percentage": 0.0,
                "realism": "indéterminé",
                "realism_details": "Analyse de secours automatique"
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
                "total_amount": 0.00,
                "conformity_rate": 0.0,
                "contestable_amount": 0.00,
                "contestable_percentage": 0.0,
                "realism": "indéterminé",
                "realism_details": "Analyse impossible"
            },
            "recommendations": [
                "L'analyse automatique a échoué. Veuillez vérifier le format de vos documents.",
                "Essayez de copier-coller directement le texte plutôt que d'utiliser des fichiers."
            ]
        }

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

def plot_charges_breakdown(charges_analysis):
    """Crée un graphique de répartition des charges"""
    if not charges_analysis:
        return None

    # Préparer les données
    descriptions = [c["description"] for c in charges_analysis]
    amounts = [c["amount"] for c in charges_analysis]

    # Si trop de catégories, regrouper les plus petites
    if len(descriptions) > 8:
        # Trier par montant
        sorted_data = sorted(zip(amounts, descriptions), reverse=True)
        top_values = sorted_data[:7]  # Garder les 7 plus grandes
        other_sum = sum(amount for amount, _ in sorted_data[7:])
        
        # Reconstruire les listes
        amounts = [amount for amount, _ in top_values] + [other_sum]
        descriptions = [desc for _, desc in top_values] + ["Autres"]

    # Graphique camembert
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        amounts, 
        labels=descriptions, 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )

    # Ajuster les propriétés du texte
    plt.setp(autotexts, size=8, weight='bold')
    plt.setp(texts, size=8)

    # Ajouter une légende
    ax.legend(
        wedges, 
        [f"{desc} ({amt:.2f}€)" for desc, amt in zip(descriptions, amounts)],
        title="Postes de charges",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=8
    )

    plt.title('Répartition des charges locatives')
    plt.tight_layout()
    
    return fig

def generate_pdf_report(analysis, bail_type, surface=None):
    """Génère un rapport PDF de l'analyse"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import cm
    from io import BytesIO
    import datetime
    import tempfile
    
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
    
    # Titre et date
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    title = f"Analyse des Charges Locatives - Bail {bail_type.capitalize()}"
    story.append(Paragraph(title, styles['Center']))
    story.append(Paragraph(f"Rapport généré le {today}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Informations générales
    story.append(Paragraph("Informations générales", styles['Heading2']))
    
    info_data = [
        ["Type de bail", bail_type.capitalize()],
        ["Surface", f"{surface} m²" if surface else "Non spécifiée"],
        ["Montant total des charges", f"{analysis['global_analysis']['total_amount']:.2f}€"],
    ]
    
    if 'charge_per_sqm' in analysis['global_analysis'] and analysis['global_analysis']['charge_per_sqm']:
        info_data.append(["Charges au m²/an", f"{analysis['global_analysis']['charge_per_sqm']:.2f}€"])
    
    info_data.append(["Taux de conformité", f"{analysis['global_analysis']['conformity_rate']:.1f}%"])
    info_data.append(["Charges contestables", f"{analysis['global_analysis'].get('contestable_percentage', 0):.1f}%"])
    info_data.append(["Réalisme", analysis['global_analysis'].get('realism', 'Non évalué')])
    
    info_table = Table(info_data, colWidths=[5*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Réalisme des charges
    if 'realism_details' in analysis['global_analysis']:
        story.append(Paragraph("Analyse du réalisme des charges", styles['Heading3']))
        story.append(Paragraph(analysis['global_analysis']['realism_details'], styles['Justify']))
        story.append(Spacer(1, 0.5*cm))
    
    # Graphique de répartition des charges
    if analysis["charges_analysis"]:
        story.append(Paragraph("Répartition des charges", styles['Heading2']))
        
        # Créer un graphique temporaire
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig_path = tmp.name
            
            # Créer le graphique
            descriptions = [c["description"] for c in analysis["charges_analysis"]]
            amounts = [c["amount"] for c in analysis["charges_analysis"]]
            
            # Limiter à 8 éléments pour la lisibilité du graphique
            if len(descriptions) > 8:
                # Trier par montant et regrouper les petites valeurs
                sorted_data = sorted(zip(amounts, descriptions), reverse=True)
                top_values = sorted_data[:7]  # Garder les 7 plus grandes
                other_sum = sum(amount for amount, _ in sorted_data[7:])
                
                # Reconstruire les listes
                amounts = [amount for amount, _ in top_values] + [other_sum]
                descriptions = [desc for _, desc in top_values] + ["Autres"]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(
                amounts, 
                labels=descriptions, 
                autopct='%1.1f%%',
                textprops={'fontsize': 9}
            )
            plt.setp(autotexts, size=8, weight='bold')
            plt.title('Répartition des charges locatives')
            plt.tight_layout()
            plt.savefig(fig_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Ajouter l'image au PDF
            img = Image(fig_path, width=15*cm, height=12*cm)
            story.append(img)
        
        story.append(Spacer(1, 0.5*cm))
    
    # Analyse détaillée des charges
    story.append(Paragraph("Analyse détaillée des charges", styles['Heading2']))
    
    charges_data = [["Description", "Montant (€)", "% du total", "Conformité", "Contestable"]]
    
    for charge in analysis["charges_analysis"]:
        charges_data.append([
            charge["description"],
            f"{charge['amount']:.2f}€",
            f"{charge['percentage']:.1f}%",
            charge["conformity"],
            "Oui" if charge.get("contestable") else "Non"
        ])
    
    # Créer un tableau pour les charges
    charges_table = Table(charges_data, colWidths=[6*cm, 2.5*cm, 2*cm, 2.5*cm, 2*cm])
    charges_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 0), (2, -1), 'RIGHT'),
    ]))
    story.append(charges_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Charges potentiellement contestables
    contestable_charges = [c for c in analysis["charges_analysis"] if c.get("contestable")]
    if contestable_charges:
        story.append(PageBreak())
        story.append(Paragraph("Charges potentiellement contestables", styles['Heading2']))
        
        for charge in contestable_charges:
            story.append(Paragraph(f"{charge['description']} ({charge['amount']:.2f}€)", styles['Heading3']))
            story.append(Paragraph(f"Montant: {charge['amount']:.2f}€ ({charge['percentage']:.1f}% du total)", styles['Normal']))
            
            if "contestable_reason" in charge and charge["contestable_reason"]:
                story.append(Paragraph(f"Raison: {charge['contestable_reason']}", styles['Normal']))
            else:
                story.append(Paragraph(f"Raison: {charge.get('conformity_details', 'Non spécifiée')}", styles['Normal']))
            
            if "matching_clause" in charge and charge["matching_clause"]:
                story.append(Paragraph("Clause correspondante dans le bail:", styles['Normal']))
                story.append(Paragraph(charge['matching_clause'], styles['Justify']))
            
            story.append(Spacer(1, 0.3*cm))
    
    # Recommandations
    story.append(Paragraph("Recommandations", styles['Heading2']))
    
    for i, rec in enumerate(analysis["recommendations"]):
        story.append(Paragraph(f"{i+1}. {rec}", styles['Normal']))
    
    story.append(Spacer(1, 0.5*cm))
    
    # Pied de page
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Ce rapport a été généré automatiquement et sert uniquement à titre indicatif. "
                          "Pour une analyse juridique complète, veuillez consulter un professionnel du droit.", 
                          styles['Normal']))
    
    # Construire le PDF
    doc.build(story)
    
    # Récupérer le contenu du buffer
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

# ----------- INTERFACE UTILISATEUR -----------

def main():
    """Fonction principale de l'application Streamlit"""
    
    st.title("Analyseur de Charges Locatives")
    st.markdown("📊 Analysez facilement la conformité de vos charges locatives avec les clauses de votre bail")
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["Saisie des documents", "Analyse des charges", "Rapport"])
    
    with tab1:
        st.header("1. Informations du bail")
        
        # Type de bail
        bail_type = st.radio("Type de bail :", ["commercial", "habitation"])
        
        # Surface
        surface = st.text_input("Surface en m² (optionnel) :")
        
        # Méthode de saisie des documents
        input_method = st.radio("Méthode de saisie :", ["Copier-coller du texte", "Téléchargement de fichiers"])
        
        bail_clauses = ""
        charges_details = ""
        
        if input_method == "Copier-coller du texte":
            # Saisie directe
            bail_clauses = st.text_area("Clauses du bail concernant les charges :", height=200, 
                                       help="Copiez-collez les clauses du bail relatives aux charges locatives")
            
            charges_details = st.text_area("Détail des charges facturées :", height=200,
                                          help="Copiez-collez le détail des charges facturées (reddition de charges)")
            
        else:
            # Téléchargement de fichiers
            st.subheader("Téléchargement du bail")
            bail_files = st.file_uploader("Téléchargez le bail (PDF, Word, image, txt)", 
                                       type=["pdf", "docx", "jpg", "jpeg", "png", "txt"], 
                                       accept_multiple_files=True)
            
            if bail_files:
                bail_clauses = process_multiple_files(bail_files)
                with st.expander("Aperçu du bail"):
                    if len(bail_files) == 1:
                        display_file_preview(bail_files[0])
                    st.text_area("Texte extrait du bail", value=bail_clauses[:1000] + "..." if len(bail_clauses) > 1000 else bail_clauses, height=150)
            
            st.subheader("Téléchargement de la reddition de charges")
            charges_files = st.file_uploader("Téléchargez la reddition de charges (PDF, Word, image, txt)", 
                                          type=["pdf", "docx", "jpg", "jpeg", "png", "txt"], 
                                          accept_multiple_files=True)
            
            if charges_files:
                charges_details = process_multiple_files(charges_files)
                with st.expander("Aperçu de la reddition de charges"):
                    if len(charges_files) == 1:
                        display_file_preview(charges_files[0])
                    st.text_area("Texte extrait de la reddition", value=charges_details[:1000] + "..." if len(charges_details) > 1000 else charges_details, height=150)
        
        # Bouton d'analyse
        if st.button("📊 Lancer l'analyse des charges"):
            if bail_clauses and charges_details:
                with st.spinner("Analyse en cours..."):
                    # Stocker les données dans la session state
                    st.session_state.bail_clauses = bail_clauses
                    st.session_state.charges_details = charges_details
                    st.session_state.bail_type = bail_type
                    st.session_state.surface = surface
                    
                    # Lancer l'analyse avec GPT-4o-mini
                    result = analyze_with_gpt4o_mini(bail_clauses, charges_details, bail_type, surface)
                    
                    # Stocker le résultat
                    st.session_state.analysis_result = result
                    st.session_state.analysis_complete = True
                    
                    # Rediriger vers l'onglet d'analyse
                    st.experimental_rerun()
            else:
                st.error("Veuillez saisir à la fois les clauses du bail et le détail des charges.")
    
    with tab2:
        st.header("2. Analyse des charges")
        
        if not st.session_state.analysis_complete:
            st.info("Veuillez d'abord compléter l'étape 1 : saisie des documents")
        else:
            result = st.session_state.analysis_result
            
            # Résumé global
            st.subheader("Résumé global")
            
            # Afficher les informations principales dans des métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total des charges", f"{result['global_analysis']['total_amount']:.2f}€")
            
            with col2:
                conformity = f"{result['global_analysis']['conformity_rate']:.1f}%"
                st.metric("Taux de conformité", conformity)
            
            with col3:
                if 'charge_per_sqm' in result['global_analysis'] and result['global_analysis']['charge_per_sqm']:
                    st.metric("Charge au m²", f"{result['global_analysis']['charge_per_sqm']:.2f}€/m²")
                else:
                    st.metric("Charges contestables", f"{result['global_analysis'].get('contestable_percentage', 0):.1f}%")
            
            # Graphique
            st.subheader("Répartition des charges")
            fig = plot_charges_breakdown(result["charges_analysis"])
            if fig:
                st.pyplot(fig)
            
            # Analyse de réalisme
            if 'realism' in result['global_analysis'] and 'realism_details' in result['global_analysis']:
                st.subheader("Analyse du réalisme")
                
                realism = result['global_analysis']['realism']
                if realism == "normal":
                    st.success(result['global_analysis']['realism_details'])
                elif realism == "bas":
                    st.info(result['global_analysis']['realism_details'])
                elif realism == "élevé":
                    st.warning(result['global_analysis']['realism_details'])
                else:
                    st.info(result['global_analysis']['realism_details'])
            
            # Tableau détaillé des charges
            st.subheader("Détail des charges")
            
            # Préparation des données pour le tableau
            charges_data = []
            for charge in result["charges_analysis"]:
                conformity = charge["conformity"]
                conformity_icon = "✅" if conformity == "conforme" else "⚠️"
                contestable_icon = "⚠️" if charge.get("contestable", False) else "✅"
                
                charges_data.append({
                    "Description": charge["description"],
                    "Montant": f"{charge['amount']:.2f}€",
                    "% du total": f"{charge['percentage']:.1f}%",
                    "Conformité": f"{conformity_icon} {conformity}",
                    "Contestable": contestable_icon
                })
            
            # Afficher le tableau
            df = pd.DataFrame(charges_data)
            st.dataframe(df, use_container_width=True)
            
            # Charges contestables
            contestable_charges = [c for c in result["charges_analysis"] if c.get("contestable", False)]
            if contestable_charges:
                st.subheader("Charges potentiellement contestables")
                
                for i, charge in enumerate(contestable_charges):
                    with st.expander(f"{charge['description']} ({charge['amount']:.2f}€ - {charge['percentage']:.1f}%)"):
                        if "contestable_reason" in charge and charge["contestable_reason"]:
                            st.warning(f"**Raison:** {charge['contestable_reason']}")
                        
                        if "matching_clause" in charge and charge["matching_clause"]:
                            st.info(f"**Clause du bail:** {charge['matching_clause']}")
                        
                        if "conformity_details" in charge:
                            st.write(f"**Détails:** {charge['conformity_details']}")
            
            # Recommandations
            st.subheader("Recommandations")
            for i, rec in enumerate(result["recommendations"]):
                st.write(f"{i+1}. {rec}")
    
    with tab3:
        st.header("3. Rapport")
        
        if not st.session_state.analysis_complete:
            st.info("Veuillez d'abord compléter l'étape 1 : saisie des documents")
        else:
            st.subheader("Générer un rapport PDF")
            
            if st.button("📄 Générer le rapport PDF"):
                with st.spinner("Génération du rapport en cours..."):
                    result = st.session_state.analysis_result
                    bail_type = st.session_state.bail_type
                    surface = st.session_state.surface
                    
                    pdf_content = generate_pdf_report(result, bail_type, surface)
                    
                    # Préparer le téléchargement
                    b64_pdf = base64.b64encode(pdf_content).decode('utf-8')
                    
                    # Créer un lien de téléchargement
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    pdf_filename = f"analyse_charges_{bail_type}_{current_date}.pdf"
                    
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">📥 Télécharger le rapport PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Afficher un aperçu
                    st.subheader("Aperçu du rapport")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
