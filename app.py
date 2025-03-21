import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from openai import OpenAI
import re

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
        "charges/m²/an": {
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
        "charges/m²/an": {
            "min": 15,
            "max": 60,
            "median": 35
        }
    }
}

# Initialisation de l'état de la session
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Fonction pour initialiser le client OpenAI
def get_openai_client():
    api_key = st.session_state.api_key
    if not api_key:
        st.error("Clé API OpenAI non configurée. Veuillez la configurer dans les paramètres.")
        return None
    return OpenAI(api_key=api_key)

# Extraction des charges avec regex (comme backup si GPT ne fonctionne pas)
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

# Appel à l'API OpenAI pour analyser les clauses et les charges
def analyze_with_openai(client, bail_clauses, charges_details, bail_type, surface=None):
    """Analyse des charges et clauses avec GPT-4o-mini"""
    try:
        # Construction du prompt pour OpenAI
        prompt = f"""
        # Analyse de charges locatives
        
        ## Contexte
        Je dois analyser des charges locatives qui me sont refacturées par mon bailleur pour vérifier leur cohérence avec les clauses du bail. Il s'agit d'un bail {bail_type}.

        ## Référentiel
        Pour un bail {bail_type}, les charges habituellement refacturables comprennent:
        {', '.join(CHARGES_TYPES[bail_type])}

        Les charges souvent contestables comprennent:
        {', '.join(CHARGES_CONTESTABLES)}

        ## Clauses du bail concernant les charges
        ```
        {bail_clauses}
        ```

        ## Détail des charges refacturées
        ```
        {charges_details}
        ```

        ## Surface locative
        {surface if surface else "Non spécifiée"}

        ## Tâche
        1. Extraire et lister toutes les clauses contractuelles définissant les charges refacturables.
        2. Extraire et lister toutes les charges refacturées avec leur montant.
        3. Pour chaque charge, analyser sa conformité avec les clauses du bail.
        4. Identifier les charges potentiellement contestables.
        5. Calculer le total des charges et, si la surface est fournie, les charges au m².
        6. Analyser le réalisme des charges par rapport aux références suivantes:
           - Bail commercial: {RATIOS_REFERENCE['commercial']['charges/m²/an']['min']}-{RATIOS_REFERENCE['commercial']['charges/m²/an']['max']}€/m²/an
           - Bail d'habitation: {RATIOS_REFERENCE['habitation']['charges/m²/an']['min']}-{RATIOS_REFERENCE['habitation']['charges/m²/an']['max']}€/m²/an
        7. Formuler des recommandations.

        ## Format de réponse
        Réponds en JSON structuré comme ceci:
        ```json
        {{
            "clauses_analysis": [
                {{
                    "title": "Titre de la clause",
                    "text": "Texte de la clause"
                }}
            ],
            "charges_analysis": [
                {{
                    "category": "Catégorie de la charge",
                    "description": "Description de la charge",
                    "amount": 1000.00,
                    "percentage": 25.0,
                    "conformity": "conforme|à vérifier",
                    "conformity_details": "Détails sur la conformité",
                    "matching_clause": "Titre de la clause correspondante ou null",
                    "contestable": true|false,
                    "contestable_reason": "Raison de la contestabilité ou null"
                }}
            ],
            "global_analysis": {{
                "total_amount": 4000.00,
                "charge_per_sqm": 100.00,
                "conformity_rate": 75.0,
                "realism": "normal|bas|élevé",
                "realism_details": "Détails sur le réalisme"
            }},
            "recommendations": [
                "Recommandation 1",
                "Recommandation 2"
            ]
        }}
        ```
        
        NE RÉPONDS QU'AVEC LE JSON, SANS AUCUN AUTRE TEXTE.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse avec OpenAI: {str(e)}")
        # Fallback avec analyse simple
        try:
            charges = extract_charges_fallback(charges_details)
            total_amount = sum(charge["amount"] for charge in charges)
            
            return {
                "clauses_analysis": [{"title": "Clause extraite manuellement", "text": clause.strip()} for clause in bail_clauses.split('\n') if clause.strip()],
                "charges_analysis": [
                    {
                        "category": charge["category"],
                        "description": charge["description"],
                        "amount": charge["amount"],
                        "percentage": (charge["amount"] / total_amount * 100) if total_amount > 0 else 0,
                        "conformity": "à vérifier",
                        "conformity_details": "Analyse de backup (OpenAI indisponible)",
                        "matching_clause": None,
                        "contestable": False,
                        "contestable_reason": None
                    } for charge in charges
                ],
                "global_analysis": {
                    "total_amount": total_amount,
                    "charge_per_sqm": total_amount / float(surface) if surface else None,
                    "conformity_rate": 0,
                    "realism": "indéterminé",
                    "realism_details": "Analyse de backup (OpenAI indisponible)"
                },
                "recommendations": [
                    "Vérifier manuellement la conformité des charges avec les clauses du bail",
                    "Demander des justificatifs détaillés pour toutes les charges importantes"
                ]
            }
        except Exception as fallback_error:
            st.error(f"Erreur lors de l'analyse de backup: {str(fallback_error)}")
            return None

def plot_charges_breakdown(charges_analysis):
    """Crée un graphique de répartition des charges"""
    if not charges_analysis:
        return None
    
    # Préparer les données
    descriptions = [c["description"] for c in charges_analysis]
    amounts = [c["amount"] for c in charges_analysis]
    
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

# Interface utilisateur Streamlit
def main():
    st.title("Analyseur de Charges Locatives avec GPT-4o-mini")
    st.markdown("""
    Cet outil utilise l'IA de OpenAI (GPT-4o-mini) pour analyser la cohérence entre les charges 
    qui vous sont refacturées par votre bailleur et les clauses de votre contrat de bail.
    """)
    
    # Sidebar pour la configuration
    st.sidebar.header("Configuration")
    
    # Configuration API OpenAI
    with st.sidebar.expander("Configuration API OpenAI", expanded=False):
        api_key = st.text_input(
            "Clé API OpenAI", 
            value=st.session_state.api_key,
            type="password",
            help="Votre clé API OpenAI (commence par 'sk-')"
        )
        st.session_state.api_key = api_key
        
        st.info("Vous pouvez obtenir une clé API sur [platform.openai.com](https://platform.openai.com/api-keys)")
    
    bail_type = st.sidebar.selectbox(
        "Type de bail",
        options=["commercial", "habitation"],
        index=0
    )
    
    surface = st.sidebar.text_input(
        "Surface locative (m²)",
        help="Utilisé pour calculer le ratio de charges au m²"
    )
    
    st.sidebar.header("Charges de référence")
    st.sidebar.markdown(f"""
    **Charges habituellement refacturables (bail {bail_type}):**
    """)
    for charge_type in CHARGES_TYPES[bail_type]:
        st.sidebar.markdown(f"- {charge_type}")
    
    st.sidebar.markdown("**Charges souvent contestables:**")
    for charge in CHARGES_CONTESTABLES:
        st.sidebar.markdown(f"- {charge}")
    
    # Formulaire principal
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clauses du bail concernant les charges")
            bail_clauses = st.text_area(
                "Copiez-collez les clauses du bail concernant les charges refacturables",
                height=250,
                help="Utilisez un format avec une clause par ligne, commençant par •, - ou un numéro"
            )
        
        with col2:
            st.subheader("Détail des charges refacturées")
            charges_details = st.text_area(
                "Entrez le détail des charges (poste et montant)",
                height=250,
                help="Format recommandé: une charge par ligne avec le montant en euros (ex: 'Nettoyage: 1200€')"
            )
        
        specific_questions = st.text_area(
            "Questions spécifiques (facultatif)",
            help="Avez-vous des questions particulières concernant certaines charges?"
        )
        
        submitted = st.form_submit_button("Analyser les charges")
    
    # Traitement du formulaire
    if submitted:
        if not bail_clauses or not charges_details:
            st.error("Veuillez remplir les champs obligatoires (clauses du bail et détail des charges).")
        elif not st.session_state.api_key:
            st.error("Veuillez configurer votre clé API OpenAI dans la barre latérale.")
        else:
            with st.spinner("Analyse en cours avec GPT-4o-mini..."):
                client = get_openai_client()
                if client:
                    # Analyser les charges avec OpenAI
                    analysis = analyze_with_openai(client, bail_clauses, charges_details, bail_type, surface)
                    if analysis:
                        st.session_state.analysis = analysis
                        st.session_state.analysis_complete = True
    
    # Afficher les résultats
    if st.session_state.analysis_complete:
        analysis = st.session_state.analysis
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
                "Montant (€)": charge["amount"],
                "% du total": f"{charge['percentage']:.1f}%",
                "Conformité": charge["conformity"],
                "Détails": charge["conformity_details"],
                "Contestable": "Oui" if charge["contestable"] else "Non"
            }
            for charge in charges_analysis
        ])
        
        # Définir le style conditionnel
        def highlight_conformity(val):
            if val == "conforme":
                return 'background-color: rgba(76, 175, 80, 0.2)'
            elif val == "à vérifier":
                return 'background-color: rgba(255, 152, 0, 0.2)'
            return ''
        
        def highlight_contestable(val):
            if val == "Oui":
                return 'background-color: rgba(244, 67, 54, 0.2)'
            return ''
        
        # Appliquer le style et afficher
        styled_df = df.style.applymap(highlight_conformity, subset=['Conformité']) \
                            .applymap(highlight_contestable, subset=['Contestable'])
        
        st.dataframe(styled_df)
        
        # Charges contestables
        contestable_charges = [c for c in charges_analysis if c.get("contestable")]
        if contestable_charges:
            st.subheader("Charges potentiellement contestables")
            for i, charge in enumerate(contestable_charges):
                with st.expander(f"{charge['description']} ({charge['amount']:.2f}€)"):
                    st.markdown(f"**Montant:** {charge['amount']:.2f}€ ({charge['percentage']:.1f}% du total)")
                    
                    if "contestable_reason" in charge and charge["contestable_reason"]:
                        st.markdown(f"**Raison:** {charge['contestable_reason']}")
                    else:
                        st.markdown(f"**Raison:** {charge['conformity_details']}")
                    
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
        
        # Export des résultats
        st.download_button(
            label="Télécharger l'analyse complète (JSON)",
            data=json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8'),
            file_name='analyse_charges.json',
            mime='application/json',
        )

if __name__ == "__main__":
    main()
