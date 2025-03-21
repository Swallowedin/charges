import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Charges Locatives",
    page_icon="📊",
    layout="wide"
)

# Initialisation NLTK (à commenter si déjà installé)
# Décommenter lors du premier déploiement
# @st.cache_resource
# def download_nltk_resources():
#     nltk.download('punkt')
#     nltk.download('stopwords')
# download_nltk_resources()

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

# Fonctions d'analyse
def extract_clauses(text):
    """Extrait les clauses du texte du bail"""
    clauses = []
    lines = text.split('\n')
    current_clause = ""
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Nouvelle clause (commence par •, -, ou un numéro)
        if line.startswith('•') or line.startswith('-') or re.match(r'^\d+\.', line):
            # Enregistrer la clause précédente si elle existe
            if current_clause:
                clauses.append({
                    "title": current_clause,
                    "text": current_text.strip()
                })
            
            # Nouvelle clause
            current_clause = re.sub(r'^[•\-\d\.]+\s*', '', line).strip()
            current_text = ""
        else:
            # Suite de la clause courante
            current_text += " " + line
    
    # Ajouter la dernière clause
    if current_clause:
        clauses.append({
            "title": current_clause,
            "text": current_text.strip()
        })
    
    return clauses

def extract_charges(text):
    """Extrait les charges du texte de la reddition"""
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

def find_matching_clauses(charge, clauses):
    """Trouve les clauses correspondant à une charge"""
    # Extraire les mots-clés de la charge
    charge_text = (charge["category"] + " " + charge["description"]).lower()
    
    # Tokeniser et filtrer les stopwords français
    french_stopwords = set(stopwords.words('french'))
    charge_tokens = [w for w in word_tokenize(charge_text, language='french') 
                    if w.isalpha() and len(w) > 2 and w not in french_stopwords]
    
    matching_clauses = []
    for clause in clauses:
        clause_text = (clause["title"] + " " + clause["text"]).lower()
        clause_tokens = [w for w in word_tokenize(clause_text, language='french') 
                        if w.isalpha() and len(w) > 2 and w not in french_stopwords]
        
        # Vérifier les correspondances de mots-clés
        matches = set(charge_tokens) & set(clause_tokens)
        if matches:
            matching_clauses.append({
                "clause": clause,
                "match_score": len(matches) / len(set(charge_tokens)),
                "matching_words": list(matches)
            })
    
    # Trier par score de correspondance
    return sorted(matching_clauses, key=lambda x: x["match_score"], reverse=True)

def analyze_charges(charges, clauses, surface=None, bail_type="commercial"):
    """Analyse complète des charges"""
    analysis_results = []
    total_amount = sum(charge["amount"] for charge in charges)
    
    for charge in charges:
        matching = find_matching_clauses(charge, clauses)
        
        result = {
            "category": charge["category"],
            "description": charge["description"],
            "amount": charge["amount"],
            "percentage": (charge["amount"] / total_amount * 100) if total_amount > 0 else 0,
            "matching_clauses": matching,
            "conformity": "conforme" if matching else "à vérifier",
            "conformity_details": ""
        }
        
        # Détails de conformité
        if matching:
            best_match = matching[0]
            result["conformity_details"] = f"Correspond à la clause '{best_match['clause']['title']}' (score: {best_match['match_score']:.2f})"
        else:
            result["conformity_details"] = "Aucune clause du bail ne semble couvrir explicitement cette charge"
        
        # Vérifier si la charge est potentiellement contestable
        result["contestable"] = is_charge_contestable(charge, total_amount, matching)
        
        analysis_results.append(result)
    
    # Analyse globale
    global_analysis = {
        "total_amount": total_amount,
        "charge_per_sqm": total_amount / float(surface) if surface else None,
        "conformity_rate": sum(1 for r in analysis_results if r["conformity"] == "conforme") / len(analysis_results) if analysis_results else 0,
        "contestable_charges": [r for r in analysis_results if r["contestable"]],
        "realism": "indéterminé"
    }
    
    # Évaluer le réalisme des charges si la surface est fournie
    if surface and float(surface) > 0:
        charge_per_sqm = total_amount / float(surface)
        ref = RATIOS_REFERENCE[bail_type]["charges/m²/an"]
        
        if charge_per_sqm < ref["min"]:
            global_analysis["realism"] = "bas"
            global_analysis["realism_details"] = f"Les charges ({charge_per_sqm:.2f}€/m²/an) sont inférieures à la fourchette habituelle ({ref['min']}-{ref['max']}€/m²/an)"
        elif charge_per_sqm > ref["max"]:
            global_analysis["realism"] = "élevé"
            global_analysis["realism_details"] = f"Les charges ({charge_per_sqm:.2f}€/m²/an) sont supérieures à la fourchette habituelle ({ref['min']}-{ref['max']}€/m²/an)"
        else:
            global_analysis["realism"] = "normal"
            global_analysis["realism_details"] = f"Les charges ({charge_per_sqm:.2f}€/m²/an) sont dans la fourchette habituelle ({ref['min']}-{ref['max']}€/m²/an)"
    
    return {
        "charges_analysis": analysis_results,
        "global_analysis": global_analysis
    }

def is_charge_contestable(charge, total_amount, matching_clauses):
    """Détermine si une charge est potentiellement contestable"""
    description = charge["description"].lower()
    
    # Pas de correspondance dans le bail
    if not matching_clauses:
        return True
    
    # Honoraires de gestion excessifs
    if ("honoraire" in description or "gestion" in description) and charge["amount"] > total_amount * 0.1:
        return True
    
    # Mots-clés suspects
    suspicious_keywords = ["réparation", "remplacement", "installation", "amélioration", "travaux", "rénovation"]
    if any(keyword in description for keyword in suspicious_keywords):
        # Vérifier si la clause correspondante limite ce type de dépense
        best_match = matching_clauses[0]["clause"]["text"].lower()
        if "article 606" in best_match or "grosses réparations" in best_match:
            return False
        if any(keyword in description and keyword not in best_match for keyword in suspicious_keywords):
            return True
    
    return False

def generate_recommendations(analysis):
    """Génère des recommandations basées sur l'analyse"""
    recommendations = []
    
    global_analysis = analysis["global_analysis"]
    charges_analysis = analysis["charges_analysis"]
    
    # Recommandations basées sur la conformité
    if global_analysis["conformity_rate"] < 0.8:
        recommendations.append("Demander au bailleur un relevé détaillé justifiant la nature des charges qui ne correspondent pas clairement aux clauses du bail")
    
    # Recommandations basées sur les charges contestables
    if global_analysis["contestable_charges"]:
        recommendations.append("Demander des justificatifs détaillés pour les charges identifiées comme contestables")
        
        # Recommendations spécifiques par type de charge contestable
        for charge in global_analysis["contestable_charges"]:
            desc = charge["description"].lower()
            if "honoraire" in desc or "gestion" in desc:
                recommendations.append(f"Vérifier le taux des honoraires de gestion ({charge['percentage']:.1f}% du total) qui semblent élevés")
            elif any(keyword in desc for keyword in ["travaux", "réparation", "remplacement"]):
                recommendations.append("Vérifier que les travaux facturés ne relèvent pas de l'article 606 du Code civil (grosses réparations)")
    
    # Recommandations basées sur le réalisme
    if global_analysis["realism"] == "élevé":
        recommendations.append("Comparer vos charges avec des immeubles similaires pour vérifier si leur niveau est justifié")
    
    # Recommandations générales
    recommendations.append("Exiger systématiquement des factures justificatives pour les charges importantes")
    recommendations.append("Vérifier l'application correcte des clés de répartition définies dans le bail")
    
    return recommendations

def plot_charges_breakdown(charges_analysis):
    """Crée un graphique de répartition des charges"""
    if not charges_analysis:
        return None
    
    # Préparer les données
    df = pd.DataFrame(charges_analysis)
    
    # Graphique camembert
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        df['amount'], 
        labels=df['description'], 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )
    
    # Ajuster les propriétés du texte
    plt.setp(autotexts, size=8, weight='bold')
    plt.setp(texts, size=8)
    
    # Ajouter une légende
    ax.legend(
        wedges, 
        [f"{row['description']} ({row['amount']:.2f}€)" for _, row in df.iterrows()],
        title="Postes de charges",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=8
    )
    
    plt.title('Répartition des charges locatives')
    plt.tight_layout()
    
    return fig

def display_conformity_table(charges_analysis):
    """Affiche un tableau des charges avec leur conformité"""
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
    
    return styled_df

# Interface utilisateur Streamlit
def main():
    st.title("Analyseur de Charges Locatives")
    st.markdown("""
    Cet outil vous aide à analyser la cohérence entre les charges qui vous sont refacturées 
    par votre bailleur et les clauses de votre contrat de bail.
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
        else:
            with st.spinner("Analyse en cours..."):
                # Extraire les données
                clauses = extract_clauses(bail_clauses)
                charges = extract_charges(charges_details)
                
                if not charges:
                    st.error("Aucune charge n'a pu être extraite. Vérifiez le format des données.")
                    return
                
                # Analyser les charges
                analysis = analyze_charges(charges, clauses, surface, bail_type)
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
            if global_analysis['charge_per_sqm']:
                st.metric("Charges au m²/an", f"{global_analysis['charge_per_sqm']:.2f}€")
        with col3:
            st.metric("Taux de conformité", f"{global_analysis['conformity_rate']*100:.0f}%")
        
        # Alerte sur le réalisme
        if global_analysis['realism'] != "indéterminé":
            color_map = {"normal": "success", "bas": "info", "élevé": "warning"}
            st.success(global_analysis['realism_details']) if global_analysis['realism'] == "normal" else \
            st.info(global_analysis['realism_details']) if global_analysis['realism'] == "bas" else \
            st.warning(global_analysis['realism_details'])
        
        # Visualisation graphique
        st.subheader("Répartition des charges")
        fig = plot_charges_breakdown(charges_analysis)
        if fig:
            st.pyplot(fig)
        
        # Tableau d'analyse détaillée
        st.subheader("Analyse détaillée des charges")
        st.dataframe(display_conformity_table(charges_analysis))
        
        # Charges contestables
        if global_analysis["contestable_charges"]:
            st.subheader("Charges potentiellement contestables")
            for i, charge in enumerate(global_analysis["contestable_charges"]):
                with st.expander(f"{charge['description']} ({charge['amount']:.2f}€)"):
                    st.markdown(f"**Montant:** {charge['amount']:.2f}€ ({charge['percentage']:.1f}% du total)")
                    st.markdown(f"**Raison:** {charge['conformity_details']}")
                    
                    if charge["matching_clauses"]:
                        best_match = charge["matching_clauses"][0]
                        st.markdown(f"""
                        **Meilleure correspondance avec le bail:**
                        >{best_match['clause']['title']}
                        
                        >*{best_match['clause']['text']}*
                        
                        **Mots-clés communs:** {', '.join(best_match['matching_words'])}
                        """)
        
        # Recommandations
        st.subheader("Recommandations")
        recommendations = generate_recommendations(analysis)
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")
        
        # Export des résultats
        st.download_button(
            label="Télécharger l'analyse complète (CSV)",
            data=pd.DataFrame(charges_analysis).to_csv(index=False).encode('utf-8'),
            file_name='analyse_charges.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
