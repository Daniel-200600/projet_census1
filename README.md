# 💰 Census Income Predictor

Une application Streamlit pour prédire si le revenu d'une personne dépasse 50K$ basé sur ses caractéristiques démographiques et économiques.

## 📊 Aperçu du Projet

Cette application utilise un modèle de Machine Learning (Random Forest) pour classifier les revenus en deux catégories :
- **Classe 0**: Revenu ≤ 50K$ (faible/moyen)
- **Classe 1**: Revenu > 50K$ (élevé)

## 🚀 Fonctionnalités

- ✅ Prédiction en temps réel du revenu
- ✅ Affichage des probabilités d'appartenance aux classes
- ✅ Interface utilisateur intuitive avec Streamlit
- ✅ Validation automatique des entrées
- ✅ Normalisation des données
- ✅ Visualisation des résultats

## 🛠️ Technologies Utilisées

- **Framework Web**: Streamlit
- **ML Framework**: scikit-learn (Random Forest)
- **Langage**: Python
- **Normalisation**: StandardScaler
- **Persistance**: Joblib

##  Prérequis

- Python 3.7+
- pip

## 🏃‍♂️ Installation et Exécution Locale

1. **Cloner le repository**:
   ```bash
   git clone https://github.com/votre-username/census-income-predictor.git
   cd census-income-predictor
   ```

2. **Installer les dépendances**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application**:
   ```bash
   streamlit run app.py
   ```

4. **Accéder à l'application**:
   Ouvrez votre navigateur à l'adresse `http://localhost:8501`

## 📁 Structure du Projet

```
census-income-predictor/
├── app.py                    # Application Streamlit principale
├── census_app.py             # Script d'analyse et entraînement du modèle
├── census.csv               # Dataset d'entraînement
├── requirements.txt          # Dépendances Python
├── .gitignore               # Fichiers à ignorer par Git
├── README.md                # Documentation
├── census.pkl              # Modèle entraîné (Random Forest)
├── scaler.pkl              # Normalisation StandardScaler
├── feature_names.pkl       # Noms des variables prédictives
└── *.png                   # Graphiques d'analyse
```

## 🔬 Variables Prédictives

Le modèle utilise 5 variables numériques :
- **Age**: Âge de la personne
- **Education-num**: Niveau d'éducation (numérique)
- **Capital-gain**: Gains en capital
- **Capital-loss**: Pertes en capital
- **Hours-per-week**: Heures travaillées par semaine

## 📈 Métriques du Modèle

- **Accuracy**: ~85%
- **F1-Score**: ~0.70
- **Type**: Classification binaire

## 🌐 Déploiement sur Streamlit Cloud

1. **Créer un repository GitHub** et pousser le code :
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Se connecter à Streamlit Cloud** :
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Se connecter avec votre compte GitHub

3. **Déployer l'application** :
   - Sélectionner le repository
   - Spécifier le fichier principal : `app.py`
   - Cliquer sur "Deploy"

4. **Configuration avancée** (optionnel) :
   - Ajouter un `packages.txt` si besoin de dépendances système
   - Configurer les secrets si nécessaire

## 📊 Analyse des Données

Le script `census_app.py` effectue une analyse complète des données :
- Analyse exploratoire (EDA)
- Comparaison de différents modèles ML
- Sélection du meilleur modèle
- Génération de graphiques et matrices de confusion

Pour relancer l'analyse :
```bash
python census_app.py
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer des améliorations via une Pull Request
- Suggérer de nouvelles fonctionnalités


## 📞 Contact

Pour toute question ou suggestion, contactez-moi via GitHub.

---

**Développé par [Daniel]**

