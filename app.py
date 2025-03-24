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
import hashlib

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Documents avec GPT-4o-mini",
    page_icon="📊",
    layout="wide"
)

# Initialisation de l'état de la session
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

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

def get_content_hash(text1, text2):
    """Génère un hash unique pour les entrées combinées"""
    combined = (text1 or "") + "|" + (text2 or "")
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def analyze_with_openai(text1, text2, document_type):
    """
    Analyse les documents avec OpenAI, avec mise en cache pour des résultats cohérents
    """
    try:
        # Générer un hash unique pour cette combinaison de documents
        content_hash = get_content_hash(text1, text2)
        
        # Vérifier si l'analyse est déjà en cache
        if content_hash in st.session_state.analysis_cache:
            st.success("Résultat récupéré du cache (analyse précédente identique)")
            return st.session_state.analysis_cache[content_hash]
        
        prompt = f"""
        # Analyse de documents
        
        ## Contexte
        Document type: {document_type}
        
        ## Document 1
        {text1[:10000]}
        
        ## Document 2
        {text2[:10000]}
        
        ## Tâche
        1. Extraire les informations clés des documents
        2. Identifier les thèmes principaux
        3. Analyser la cohérence entre les documents
        4. Formuler des observations pertinentes
        
        ## Format JSON
        {{
            "document1_analysis":[{{"title":"","content":""}}],
            "document2_analysis":[{{"title":"","content":""}}],
            "themes": [""],
            "coherence_analysis": {{"coherence_level":"élevée|moyenne|faible","details":""}},
            "observations": [""]
        }}
        """

        # Essayer d'abord avec gpt-4o-mini
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Réduire la température pour plus de cohérence
                seed=42,  # Assurer la cohérence des résultats
                response_format={"type": "json_object"}  # Forcer une réponse JSON
            )
            result = json.loads(response.choices[0].message.content)
            st.success("Analyse réalisée avec gpt-4o-mini")
            
        except Exception as e:
            st.warning(f"Erreur avec gpt-4o-mini: {str(e)}. Tentative avec gpt-3.5-turbo...")
            
            # Si échec, basculer vers gpt-3.5-turbo
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Réduire la température pour plus de cohérence
                seed=42,  # Assurer la cohérence des résultats
                response_format={"type": "json_object"}  # Forcer une réponse JSON
            )
            
            result = json.loads(response.choices[0].message.content)
            st.success("Analyse réalisée avec gpt-3.5-turbo")
        
        # Stocker le résultat dans le cache
        st.session_state.analysis_cache[content_hash] = result
        return result

    except Exception as e:
        st.error(f"Erreur lors de l'analyse avec OpenAI: {str(e)}")
        # Retourner une analyse par défaut en cas d'erreur
        return {
            "document1_analysis": [{"title": "Analyse manuelle nécessaire", "content": "Une erreur s'est produite lors de l'analyse automatique."}],
            "document2_analysis": [{"title": "Analyse manuelle nécessaire", "content": "Une erreur s'est produite lors de l'analyse automatique."}],
            "themes": ["Non disponible suite à une erreur"],
            "coherence_analysis": {"coherence_level": "indéterminée", "details": "L'analyse n'a pas pu être effectuée automatiquement."},
            "observations": ["Veuillez réessayer ou effectuer une analyse manuelle."]
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
    Génère un rapport PDF complet de l'analyse des documents.
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
    title = f"Analyse de Documents - Type {document_type.capitalize()}"
    story.append(Paragraph(title, styles['Center']))
    story.append(Paragraph(f"Rapport généré le {today}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Informations générales
    story.append(Paragraph("Informations générales", styles['Heading2']))
    
    info_data = [
        ["Type de document", document_type.capitalize()],
        ["Niveau de cohérence", analysis['coherence_analysis']['coherence_level']]
    ]
    
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
    
    # Analyse de cohérence
    if 'details' in analysis['coherence_analysis']:
        story.append(Paragraph("Analyse de cohérence", styles['Heading3']))
        story.append(Paragraph(analysis['coherence_analysis']['details'], styles['Justify']))
        story.append(Spacer(1, 0.5*cm))
    
    # Thèmes principaux
    if analysis["themes"]:
        story.append(Paragraph("Thèmes principaux", styles['Heading2']))
        for theme in analysis["themes"]:
            story.append(Paragraph(f"• {theme}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    # Analyse du document 1
    story.append(Paragraph("Analyse du Document 1", styles['Heading2']))
    for section in analysis["document1_analysis"]:
        story.append(Paragraph(section["title"], styles['Heading3']))
        story.append(Paragraph(section["content"], styles['Justify']))
        story.append(Spacer(1, 0.3*cm))
    
    # Analyse du document 2
    story.append(PageBreak())
    story.append(Paragraph("Analyse du Document 2", styles['Heading2']))
    for section in analysis["document2_analysis"]:
        story.append(Paragraph(section["title"], styles['Heading3']))
        story.append(Paragraph(section["content"], styles['Justify']))
        story.append(Spacer(1, 0.3*cm))
    
    # Observations
    story.append(Paragraph("Observations", styles['Heading2']))
    for i, obs in enumerate(analysis["observations"]):
        story.append(Paragraph(f"{i+1}. {obs}", styles['Normal']))
    
    story.append(Spacer(1, 0.5*cm))
    
    # Pied de page
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Ce rapport a été généré automatiquement et sert uniquement à titre indicatif. "
                          "Pour une analyse complète, veuillez consulter un professionnel du domaine concerné.", 
                          styles['Small']))
    
    # Construire le PDF
    doc.build(story)
    
    # Récupérer le contenu du buffer
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

# Interface utilisateur Streamlit
def main():
    st.title("Analyseur de Documents avec GPT-4o-mini")
    st.markdown("""
    Cet outil analyse la cohérence entre deux documents en utilisant GPT-4o-mini.
    Pour les mêmes documents en entrée, l'analyse sera identique à chaque exécution.
    """)

    # Sidebar pour la configuration
    st.sidebar.header("Configuration")

    document_type = st.sidebar.selectbox(
        "Type de document",
        options=["rapport", "contrat", "article", "correspondance", "autre"],
        index=0
    )

    # Interface principale avec onglets
    tab1, tab2 = st.tabs(["Saisie manuelle", "Téléchargement de fichiers"])

    # Onglet 1: Saisie manuelle
    with tab1:
        with st.form("input_form_manual"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Document 1")
                document1_manual = st.text_area(
                    "Copiez-collez le contenu du premier document",
                    height=250
                )

            with col2:
                st.subheader("Document 2")
                document2_manual = st.text_area(
                    "Copiez-collez le contenu du deuxième document",
                    height=250
                )

            specific_questions = st.text_area(
                "Questions spécifiques (facultatif)",
                help="Avez-vous des questions particulières concernant ces documents?"
            )

            submitted_manual = st.form_submit_button("Analyser les documents")

    # Onglet 2: Téléchargement de fichiers
    with tab2:
        with st.form("input_form_files"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Document 1")
                doc1_files = st.file_uploader(
                    "Téléchargez le(s) fichier(s) du document 1 (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True
                )

                if doc1_files:
                    st.write(f"{len(doc1_files)} fichier(s) téléchargé(s) pour le document 1")

            with col2:
                st.subheader("Document 2")
                doc2_files = st.file_uploader(
                    "Téléchargez le(s) fichier(s) du document 2 (PDF, Word, TXT, Image)",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    accept_multiple_files=True
                )

                if doc2_files:
                    st.write(f"{len(doc2_files)} fichier(s) téléchargé(s) pour le document 2")

            specific_questions_file = st.text_area(
                "Questions spécifiques (facultatif)",
                help="Avez-vous des questions particulières concernant ces documents?"
            )

            submitted_files = st.form_submit_button("Analyser les fichiers")

    # Traitement du formulaire de saisie manuelle
    if submitted_manual:
        if not document1_manual or not document2_manual:
            st.error("Veuillez remplir les champs obligatoires (document 1 et document 2).")
        else:
            with st.spinner("Analyse en cours..."):
                # Analyser les documents avec OpenAI
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
            st.error("Veuillez télécharger au moins un fichier pour chaque document.")
        else:
            with st.spinner("Extraction et analyse des fichiers en cours..."):
                # Extraire et combiner le texte de tous les fichiers
                document1_combined = process_multiple_files(doc1_files)
                document2_combined = process_multiple_files(doc2_files)

                if not document1_combined or not document2_combined:
                    st.error("Impossible d'extraire le texte des fichiers téléchargés.")
                else:
                    # Afficher un résumé du texte extrait
                    st.info(f"Texte extrait: Document 1 ({len(document1_combined)} caractères), Document 2 ({len(document2_combined)} caractères)")

                    # Analyser les documents avec OpenAI
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

        st.header("Résultats de l'analyse")

        # Afficher le niveau de cohérence
        coherence_level = analysis['coherence_analysis']['coherence_level']
        coherence_details = analysis['coherence_analysis']['details']
        
        # Définir la couleur en fonction du niveau de cohérence
        color_map = {"élevée": "success", "moyenne": "warning", "faible": "error"}
        alert_type = color_map.get(coherence_level, "info")
        
        if alert_type == "success":
            st.success(f"Niveau de cohérence: {coherence_level}. {coherence_details}")
        elif alert_type == "warning":
            st.warning(f"Niveau de cohérence: {coherence_level}. {coherence_details}")
        elif alert_type == "error":
            st.error(f"Niveau de cohérence: {coherence_level}. {coherence_details}")
        else:
            st.info(f"Niveau de cohérence: {coherence_level}. {coherence_details}")

        # Afficher les thèmes principaux
        st.subheader("Thèmes principaux")
        for theme in analysis["themes"]:
            st.markdown(f"- {theme}")

        # Visualisation graphique des thèmes
        if len(analysis["themes"]) > 1:
            st.subheader("Visualisation des thèmes")
            fig = plot_themes_chart(analysis["themes"])
            if fig:
                st.pyplot(fig)

        # Afficher l'analyse des documents dans des onglets
        doc1_tab, doc2_tab = st.tabs(["Analyse Document 1", "Analyse Document 2"])
        
        with doc1_tab:
            for section in analysis["document1_analysis"]:
                with st.expander(section["title"]):
                    st.markdown(section["content"])
        
        with doc2_tab:
            for section in analysis["document2_analysis"]:
                with st.expander(section["title"]):
                    st.markdown(section["content"])

        # Observations
        st.subheader("Observations")
        for i, obs in enumerate(analysis["observations"]):
            st.markdown(f"{i+1}. {obs}")

        # Options d'export
        st.header("Exporter les résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            st.download_button(
                label="Télécharger l'analyse en JSON",
                data=json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8'),
                file_name='analyse_documents.json',
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
                    file_name="rapport_analyse_documents.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF: {str(e)}")
                st.info("Assurez-vous d'avoir installé reportlab avec 'pip install reportlab'")

if __name__ == "__main__":
    main()
