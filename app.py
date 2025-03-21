import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
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

# Récupération de la clé API depuis les secrets de Streamlit Cloud
def get_openai_client():
    try:
        api_key = st.secrets["openai"]["api_key"]
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Erreur lors de la récupération de la clé API OpenAI: {str(e)}")
        return None

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
    if not client:
        return None
        
    try:
        # Construction du prompt pour OpenAI (réduit pour optimisation)
        prompt = f"""
        # Analyse de charges locatives
        
        ## Contexte
        Bail {bail_type}, analyse des charges refacturées vs clauses du bail.

        ## Référentiel
        Charges habituellement refacturables: {', '.join(CHARGES_TYPES[bail_type])}
        Charges contestables: {', '.join(CHARGES_CONTESTABLES)}

        ## Clauses du bail
        {bail_clauses}

        ## Charges refacturées
        {charges_details}

        ## Surface: {surface if surface else "Non spécifiée"}

        ## Tâche
        1. Extraire clauses et charges avec montants
        2. Analyser conformité de chaque charge avec le bail
        3. Identifier charges contestables
        4. Calculer total et ratio/m² si surface fournie
        5. Analyser réalisme: Commercial {RATIOS_REFERENCE['commercial']['charges/m²/an']['min']}-{RATIOS_REFERENCE['commercial']['charges/m²/an']['max']}€/m²/an, Habitation {RATIOS_REFERENCE['habitation']['charges/m²/an']['min']}-{RATIOS_REFERENCE['habitation']['charges/m²/an']['max']}€/m²/an
        6. Formuler recommandations

        ## Format JSON
        {"clauses_analysis":[{"title":"","text":""}],"charges_analysis":[{"category":"","description":"","amount":0,"percentage":0,"conformity":"conforme|à vérifier","conformity_details":"","matching_clause":"","contestable":true|false,"contestable_reason":""}],"global_analysis":{"total_amount":0,"charge_per_sqm":0,"conformity_rate":0,"realism":"normal|bas|élevé","realism_details":""},"recommendations":[""]}

        NE RÉPONDS QU'AVEC LE JSON, SANS AUCUN AUTRE TEXTE.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3,  # Valeur plus basse pour réponses plus cohérentes
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
    Cet outil analyse la cohérence entre les charges refacturées par votre bailleur 
    et les clauses de votre contrat de bail en utilisant GPT-4o-mini.
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
        
        # Afficher le DataFrame sans style conditionnel pour éviter les problèmes
        st.dataframe(df)
        
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
