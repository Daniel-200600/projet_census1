# -*- coding: utf-8 -*-
"""
Census Income Prediction App - Streamlit Interface
Application de prédiction du revenu basée sur les données du Census
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from scipy import stats
import sys

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Census Income Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration UTF-8 pour les caractères spéciaux
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stdin.encoding != 'utf-8':
    sys.stdin.reconfigure(encoding='utf-8')

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

@st.cache_data
def load_data():
    """Charger les données du census"""
    return pd.read_csv('census.csv', encoding='utf-8')

# Fichiers du modèle sauvegardé (écrits par la page « Entraînement du modèle »)
MODEL_FILE = 'census.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'feature_names.pkl'

# Origine du modèle utilisé par la page Prédiction : "fichiers" ou "automatique"
SOURCE_MODELE = {"origine": None}


def train_default_model():
    """Entraîne un Random Forest par défaut sur census.csv (sans toucher au disque)."""
    df = load_data()
    X, y, numeric_cols = prepare_data(df)
    scaler = preprocessing.StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    return model, scaler, numeric_cols


@st.cache_resource
def load_model():
    """Charger le modèle sauvegardé ; à défaut (fichiers absents, ou illisibles car
    créés avec une autre version de scikit-learn), entraîner un modèle par défaut."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        feature_names = joblib.load(FEATURES_FILE)
        SOURCE_MODELE["origine"] = "fichiers"
        return model, scaler, feature_names
    except Exception:
        # FileNotFoundError, ModuleNotFoundError, erreurs de désérialisation, etc.
        try:
            SOURCE_MODELE["origine"] = "automatique"
            return train_default_model()
        except Exception:
            SOURCE_MODELE["origine"] = None
            return None, None, None

def prepare_data(df):
    """Préparer les données pour l'entraînement"""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Conversion de la variable 'income' en classification binaire
    df_clean['income'] = (df_clean['income'].str.strip() == '>50K').astype(int)
    
    X = df_clean[numeric_cols].copy()
    y = df_clean['income'].copy()
    
    return X, y, numeric_cols

def train_models(X_train, y_train, X_test, y_test):
    """Entraîner tous les modèles et retourner les résultats"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results.append({
            'Model': name,
            'model_object': model,
            'Train Accuracy': accuracy_score(y_train, model.predict(X_train)),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    
    return results

# ============================================
# BARRE LATÉRALE - NAVIGATION
# ============================================

st.sidebar.title("🏠 Menu Principal")
st.sidebar.markdown("---")

# Image/logo
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px;">
    <h2>📊 Census Predictor</h2>
    <p>Prédiction du revenu</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📈 Exploration des données", "🔧 Entraînement du modèle", "🎯 Prédiction"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**À propos:**
Cette application permet d'analyser les données du Census et de prédire si le revenu d'une personne dépasse 50K$/an.
""")

# ============================================
# PAGE: ACCUEIL
# ============================================

if page == "🏠 Accueil":
    st.title("🏠 Application de Prédiction du Revenu Census")
    st.markdown("---")
    
    # Colonnes pour la mise en page
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bienvenue ! 👋
        
        Cette application Streamlit permet de:
        
        - **Explorer** les données du Census américain
        - **Entraîner** différents modèles de Machine Learning
        - **Prédire** le revenu d'une personne basé sur ses caractéristiques
        
        ### Comment utiliser l'application:
        
        1. 📈 **Exploration des données**: Visualisez les statistiques et graphiques
        2. 🔧 **Entraînement**: Comparez différents modèles ML
        3. 🎯 **Prédiction**: Entrez les caractéristiques d'une personne
        """)
        
        # Statistiques rapides
        df = load_data()
        st.markdown("### 📊 Aperçu des données")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Nombre de lignes", f"{df.shape[0]:,}")
        with col_b:
            st.metric("Nombre de colonnes", df.shape[1])
        with col_c:
            st.metric("Colonnes numériques", len(df.select_dtypes(include=[np.number]).columns))
    
    with col2:
        st.markdown("### 🔑 Variables clés")
        st.info("""
        **Features utilisés:**
        - Age
        - Education-num
        - Capital-gain
        - Capital-loss
        - Hours-per-week
        
        **Target:**
        - Income (<=50K / >50K)
        """)
    
    # Aperçu des données
    st.markdown("---")
    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.markdown("*Utilisez la barre latérale pour naviguer entre les différentes sections*")

# ============================================
# PAGE: EXPLORATION DES DONNÉES
# ============================================

elif page == "📈 Exploration des données":
    st.title("📈 Exploration des Données")
    st.markdown("---")
    
    df = load_data()
    
    # Onglets pour différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Statistiques générales", 
        "📉 Distributions", 
        "🔗 Corrélations",
        "📋 Données"
    ])
    
    with tab1:
        st.subheader("Statistiques générales du dataset")
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Instances", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Valeurs manquantes", df.isnull().sum().sum())
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Variables numériques", len(numeric_cols))
        
    
    with tab2:
        st.subheader("Distribution des variables numériques")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_select = st.selectbox("Sélectionner une variable", numeric_cols)
        
        # Histogramme
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[col_select].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(col_select)
        ax.set_ylabel('Fréquence')
        ax.set_title(f'Distribution de {col_select}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Boxplot - Version améliorée avec gestion des valeurs aberrantes
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        
        # Récupérer les données sans valeurs nulles
        data = df[col_select].dropna()
        
        # Vérifier si la colonne a assez de données
        if len(data) > 0:
            # Utiliser seaborn pour une meilleure visualisation
            sns.boxplot(y=data, ax=ax2, color='steelblue', fliersize=5)
            ax2.set_ylabel(col_select)
            ax2.set_title(f'Boîte à moustaches de {col_select}')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Ajouter des statistiques
            stats_text = f"Min: {data.min():.2f}\nQ1: {data.quantile(0.25):.2f}\nMédiane: {data.median():.2f}\nQ3: {data.quantile(0.75):.2f}\nMax: {data.max():.2f}"
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'Aucune donnée disponible', ha='center', va='center', transform=ax2.transAxes)
        
        st.pyplot(fig2)
        
        # Afficher les valeurs aberrantes détectées
        with st.expander("Voir les valeurs aberrantes"):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            st.write(f"Nombre de valeurs aberrantes: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
            st.write(f"Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Tableau complet des statistiques
        st.markdown("### 📊 Tableau des statistiques complètes")
        
        # Calculer toutes les statistiques
        stats_dict = {
            'Statistique': [],
            'Valeur': []
        }
        
        # Nombre de valeurs non-nulles
        stats_dict['Statistique'].append('Count (Non-null)')
        stats_dict['Valeur'].append(len(data))
        
        # Moyenne
        stats_dict['Statistique'].append('Mean (Moyenne)')
        stats_dict['Valeur'].append(round(data.mean(), 4))
        
        # Écart-type
        stats_dict['Statistique'].append('Std (Écart-type)')
        stats_dict['Valeur'].append(round(data.std(), 4))
        
        # Minimum
        stats_dict['Statistique'].append('Min (Minimum)')
        stats_dict['Valeur'].append(round(data.min(), 4))
        
        # Percentiles
        for p in [5, 10, 25, 50, 75, 90, 95]:
            stats_dict['Statistique'].append(f'Percentile {p}%')
            stats_dict['Valeur'].append(round(data.quantile(p/100), 4))
        
        # Maximum
        stats_dict['Statistique'].append('Max (Maximum)')
        stats_dict['Valeur'].append(round(data.max(), 4))
        
        # Variance
        stats_dict['Statistique'].append('Variance')
        stats_dict['Valeur'].append(round(data.var(), 4))
        
        # Range (Étendue)
        stats_dict['Statistique'].append('Range (Étendue)')
        stats_dict['Valeur'].append(round(data.max() - data.min(), 4))
        
        # IQR (Interquartile Range)
        stats_dict['Statistique'].append('IQR (Étendue interquartile)')
        stats_dict['Valeur'].append(round(data.quantile(0.75) - data.quantile(0.25), 4))
        
        # Coefficient de variation
        if data.mean() != 0:
            cv = (data.std() / abs(data.mean())) * 100
            stats_dict['Statistique'].append('Coefficient de Variation (%)')
            stats_dict['Valeur'].append(round(cv, 4))
        
        # Skewness (Asymétrie)
        stats_dict['Statistique'].append('Skewness (Asymétrie)')
        stats_dict['Valeur'].append(round(data.skew(), 4))
        
        # Kurtosis (Kurtose)
        stats_dict['Statistique'].append('Kurtosis (Kurtose)')
        stats_dict['Valeur'].append(round(data.kurtosis(), 4))
        
        # Médiane
        stats_dict['Statistique'].append('Median (Médiane)')
        stats_dict['Valeur'].append(round(data.median(), 4))
        
        # Mode (valeur la plus fréquente)
        stats_dict['Statistique'].append('Mode')
        stats_dict['Valeur'].append(data.mode().iloc[0] if len(data.mode()) > 0 else 'N/A')
        
        # Somme
        stats_dict['Statistique'].append('Sum (Somme)')
        stats_dict['Valeur'].append(round(data.sum(), 4))
        
        # Créer le DataFrame des statistiques
        stats_df = pd.DataFrame(stats_dict)
        
        # Afficher le tableau avec mise en forme
        st.dataframe(
            stats_df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'border': '1px solid #ddd',
                'padding': '8px'
            }),
            use_container_width=True,
            height=600
        )
        
        # Option pour télécharger les statistiques
        csv = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les statistiques (CSV)",
            data=csv,
            file_name=f'statistics_{col_select}.csv',
            mime='text/csv',
        )
        
        # Distribution de la variable cible
        st.markdown("### Distribution de la cible (Income)")
        income_dist = df['income'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            colors = ['#ff6b6b', '#4ecdc4']
            ax3.pie(income_dist.values, labels=income_dist.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90, explode=(0.05, 0.05))
            ax3.set_title('Distribution du revenu')
            st.pyplot(fig3)
        
        with col2:
            st.markdown("### Détails de la distribution")
            st.write(f"- **<=50K:** {income_dist.iloc[0]:,} instances ({income_dist.iloc[0]/len(df)*100:.1f}%)")
            st.write(f"- **>50K:** {income_dist.iloc[1]:,} instances ({income_dist.iloc[1]/len(df)*100:.1f}%)")
    
    with tab3:
        st.subheader("Matrice de corrélation")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Matrice de corrélation
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', ax=ax, linewidths=0.5)
        ax.set_title('Matrice de corrélation des variables numériques')
        st.pyplot(fig)
        
        # Analyse des corrélations
        st.markdown("### Analyse des corrélations")
        
        # Trier les corrélations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Corrélation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Corrélation', key=abs, ascending=False)
        
        st.dataframe(corr_df.head(10), use_container_width=True)
    
    with tab4:
        st.subheader("Visualisation des données brutes")
        
        # Nombre de lignes à afficher
        n_rows = st.slider("Nombre de lignes", 5, 50, 10)
        
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        st.markdown("### Filtrer les données")
        
        # Filtres
        col1, col2 = st.columns(2)
        
        with col1:
            filter_col = st.selectbox("Colonne à filtrer", df.columns)
        
        with col2:
            if df[filter_col].dtype == 'object':
                filter_val = st.selectbox("Valeur", df[filter_col].unique())
                filtered_df = df[df[filter_col] == filter_val]
            else:
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                filter_range = st.slider("Plage de valeurs", min_val, max_val, (min_val, max_val))
                filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
        
        st.markdown(f"**{len(filtered_df)}** lignes correspondent au filtre")
        st.dataframe(filtered_df.head(20), use_container_width=True)

# ============================================
# PAGE: ENTRAÎNEMENT DU MODÈLE
# ============================================

elif page == "🔧 Entraînement du modèle":
    st.title("🔧 Entraînement des Modèles")
    st.markdown("---")
    
    # Charger les données
    df = load_data()
    X, y, numeric_cols = prepare_data(df)
    
    # Configuration de l'entraînement
    st.sidebar.markdown("### ⚙️ Configuration")
    
    test_size = st.sidebar.slider("Taille du test set (%)", 10, 40, 30) / 100
    random_state = st.sidebar.number_input("Random State", value=42)
    
    # Bouton pour lancer l'entraînement
    if st.button("🚀 Lancer l'entraînement", type="primary"):
        with st.spinner("Entraînement en cours..."):
            # Préparation des données
            scaler = preprocessing.StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Entraîner les modèles
            results = train_models(X_train, y_train, X_test, y_test)
            
            # Afficher les résultats
            st.success("Entraînement terminé !")
            
            # Tableau des résultats
            st.markdown("## 📊 Résultats des modèles")
            
            results_df = pd.DataFrame(results)[['Model', 'Train Accuracy', 'Test Accuracy', 'F1-Score']]
            results_df = results_df.sort_values('Test Accuracy', ascending=False)
            
            # Mise en forme du tableau
            st.dataframe(
                results_df.style.background_gradient(subset=['Test Accuracy', 'F1-Score'], cmap='Greens'),
                use_container_width=True
            )
            
            # Graphique comparatif
            st.markdown("### 📈 Comparaison visuelle")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy
            ax1 = axes[0]
            x_pos = np.arange(len(results_df))
            width = 0.35
            ax1.bar(x_pos - width/2, results_df['Train Accuracy'], width, label='Train', alpha=0.8)
            ax1.bar(x_pos + width/2, results_df['Test Accuracy'], width, label='Test', alpha=0.8)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy: Train vs Test')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # F1-Score
            ax2 = axes[1]
            ax2.bar(x_pos, results_df['F1-Score'], color='green', alpha=0.7)
            ax2.set_ylabel('F1-Score')
            ax2.set_title('F1-Score par modèle')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Meilleur modèle
            best_idx = results_df['Test Accuracy'].idxmax()
            best_model_name = results_df.loc[best_idx, 'Model']
            
            st.markdown(f"### 🏆 Meilleur modèle: **{best_model_name}**")

            # Sauvegarde du meilleur modèle, du scaler et des noms de variables :
            # la page Prédiction les recharge (voir load_model)
            try:
                joblib.dump(results[best_idx]['model_object'], MODEL_FILE)
                joblib.dump(scaler, SCALER_FILE)
                joblib.dump(list(numeric_cols), FEATURES_FILE)
                load_model.clear()
                st.info(f"💾 Modèle « {best_model_name} » sauvegardé : il sera utilisé par la page Prédiction.")
            except OSError as e:
                st.warning(f"Le modèle n'a pas pu être sauvegardé : {e}")
            
            # Matrice de confusion du meilleur modèle
            best_result = results[best_idx]
            y_pred = best_result['model_object'].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig2, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            ax.set_ylabel('Vraie classe')
            ax.set_xlabel('Classe prédite')
            ax.set_title(f'Matrice de Confusion - {best_model_name}')
            st.pyplot(fig2)
            
            # Rapport de classification
            st.markdown("### 📋 Rapport de classification")
            report = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
            st.text(report)
    
    # Information sur les paramètres
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Paramètres d'entraînement
    
    - **Test Size**: Pourcentage des données utilisées pour le test
    - **Random State**: Graine pour la reproductibilité
    """)

# ============================================
# PAGE: PRÉDICTION
# ============================================

elif page == "🎯 Prédiction":
    st.title("🎯 Prédiction du Revenu")
    st.markdown("---")
    
    # Charger le modèle
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.error("❌ Aucun modèle disponible. Veuillez d'abord entraîner un modèle dans l'onglet 'Entraînement du modèle'")
    else:
        if SOURCE_MODELE["origine"] == "automatique":
            st.caption("ℹ️ Modèle Random Forest entraîné automatiquement (aucun modèle sauvegardé "
                       "utilisable). Lancez l'entraînement pour en sauvegarder un.")
        # Formulaire de saisie
        st.markdown("### Entrez les caractéristiques de la personne:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Âge (18-100)", min_value=18, max_value=100, value=35)
            education_num = st.slider("Niveau d'éducation (1-16)", min_value=1, max_value=16, value=10)
            capital_gain = st.number_input("Gain en capital ($)", min_value=0, max_value=100000, value=0, step=100)
        
        with col2:
            capital_loss = st.number_input("Perte en capital ($)", min_value=0, max_value=10000, value=0, step=50)
            hours_per_week = st.slider("Heures travaillees/semaine", min_value=1, max_value=100, value=40)
        
        # Bouton de prédiction
        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            # Créer le DataFrame d'entrée
            user_input = pd.DataFrame({
                'age': [age],
                'education-num': [education_num],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week]
            })
            
            # Normaliser les données
            user_input_scaled = scaler.transform(user_input)
            
            # Prédiction
            prediction = model.predict(user_input_scaled)[0]
            
            # Probabilités si disponibles
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(user_input_scaled)[0]
            else:
                proba = None
            
            # Afficher le résultat
            st.markdown("---")
            st.markdown("### 📊 Résultat de la prédiction")
            
            if prediction == 1:
                st.success("💰 **Le revenu prédit est SUPÉRIEUR à 50K$/an (>50K)**")
            else:
                st.info("📉 **Le revenu prédit est INFÉRIEUR ou ÉGAL à 50K$/an (<=50K)**")
            
            # Afficher les probabilités
            if proba is not None:
                st.markdown("### Probabilités:")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Probabilité <=50K", f"{proba[0]*100:.1f}%")
                with col_p2:
                    st.metric("Probabilité >50K", f"{proba[1]*100:.1f}%")
                
                # Barre de progression
                st.progress(proba[1], text=f"Confiance: {max(proba)*100:.1f}%")
            
            # Récapitulatif des entrées
            st.markdown("---")
            st.markdown("### 📋 Récapitulatif des entrées:")
            st.table(user_input.T)
        
        # Section: Exemples de prédictions
        st.markdown("---")
        st.markdown("### 💡 Exemples de prédictions")
        
        # Charger des exemples
        df = load_data()
        X, y, numeric_cols = prepare_data(df)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=numeric_cols)
        
        # Prédire quelques exemples
        sample_indices = np.random.choice(len(X_scaled), 5, replace=False)
        
        examples_results = []
        for idx in sample_indices:
            sample = X_scaled.iloc[idx:idx+1]
            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None
            
            examples_results.append({
                'Index': idx,
                'Âge': df.iloc[idx]['age'],
                'Education': df.iloc[idx]['education-num'],
                'Capital Gain': df.iloc[idx]['capital-gain'],
                'Heures/semaine': df.iloc[idx]['hours-per-week'],
                'Prédiction': '>50K' if pred == 1 else '<=50K',
                'Confiance': f"{max(proba)*100:.1f}%" if proba is not None else 'N/A'
            })
        
        examples_df = pd.DataFrame(examples_results)
        st.dataframe(examples_df, use_container_width=True)
        
        st.info("💡 Ce sont des exemples aléatoires du dataset pour illustrer les prédictions du modèle.")

# ============================================
# PIED DE PAGE
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 10px;">
    <p>📊 Census Income Predictor - Application Streamlit</p>
    <p>Développé avec  menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)
