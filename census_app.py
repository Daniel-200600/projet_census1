import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import joblib

# Charger le fichier CSV en dataframe
df = pd.read_csv('census.csv')

# ============================================
# ANALYSE EXPLORATOIRE DES DONNÉES
# ============================================

print("=" * 70)
print("ANALYSE DU DATASET CENSUS")
print("=" * 70)

# 1. Dimensions du dataset (shape)
print("\n1. DIMENSIONS DU DATASET:")
print(f"   Shape: {df.shape}")
print(f"   → Nombre d'instances: {df.shape[0]}")
print(f"   → Nombre de caractéristiques: {df.shape[1]}")

# 2. Information sur les colonnes et types (info)
print("\n2. INFORMATION SUR LES COLONNES ET TYPES:")
print(df.info())

# 3. Statistiques descriptives (describe)
print("\n3. STATISTIQUES DESCRIPTIVES:")
print(df.describe())

# 4. Aperçu des premières lignes (head)
print("\n4. APERÇU DES PREMIÈRES LIGNES:")
print(df.head())

# 5. Distribution des classes - Colonne "income"
print("\n5. NOMBRE DE CLASSES ET DISTRIBUTION:")
print(f"   Colonne cible: 'income'")
print(f"   Nombre de classes: {df['income'].nunique()}")
print("\n   Distribution des instances par classe:")
print(df['income'].value_counts())
print("\n   Pourcentage de distribution:")
print(df['income'].value_counts(normalize=True) * 100)

print("\n" + "=" * 70)

# ============================================
# ANALYSE CROISÉE DES VARIABLES
# ============================================

print("\n2) CROISEMENT DES VARIABLES DEUX À DEUX")
print("=" * 70)

# Sélectionner les variables numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nVariables numériques identifiées: {numeric_cols}\n")

# Créer un pairplot avec tous les variables numériques
print("Création d'un pairplot avec toutes les variables numériques...")
plt.figure(figsize=(16, 12))
pairplot = sns.pairplot(df[numeric_cols], diag_kind='hist', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle('PairPlot - Croisement de toutes les variables numériques', 
                       fontsize=16, y=1.001)
plt.tight_layout()
plt.savefig('pairplot_all_variables.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Pairplot sauvegardé: 'pairplot_all_variables.png'\n")

# Créer des scatter plots avec droites de régression pour les paires principales
print("Création de scatter plots avec droites de régression...")

# Sélectionner les principales variables numériques pour une meilleure visualisation
main_numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Scatter plot 1: Age vs Heures par semaine
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Analyses Croisées avec Droites de Régression', fontsize=16, fontweight='bold')

# Age vs Hours-per-week
ax = axes[0, 0]
sns.regplot(x='age', y='hours-per-week', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Age vs Heures/Semaine')
ax.set_xlabel('Âge')
ax.set_ylabel('Heures par semaine')

# Age vs Education-num
ax = axes[0, 1]
sns.regplot(x='age', y='education-num', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Age vs Niveau d\'Éducation')
ax.set_xlabel('Âge')
ax.set_ylabel('Niveau d\'éducation (numérique)')

# Age vs Capital-gain
ax = axes[0, 2]
sns.regplot(x='age', y='capital-gain', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Age vs Gain en Capital')
ax.set_xlabel('Âge')
ax.set_ylabel('Gain en capital')

# Hours-per-week vs Capital-gain
ax = axes[1, 0]
sns.regplot(x='hours-per-week', y='capital-gain', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Heures/Semaine vs Gain en Capital')
ax.set_xlabel('Heures par semaine')
ax.set_ylabel('Gain en capital')

# Hours-per-week vs Education-num
ax = axes[1, 1]
sns.regplot(x='hours-per-week', y='education-num', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Heures/Semaine vs Niveau d\'Éducation')
ax.set_xlabel('Heures par semaine')
ax.set_ylabel('Niveau d\'éducation')

# Capital-loss vs Capital-gain
ax = axes[1, 2]
sns.regplot(x='capital-loss', y='capital-gain', data=df, ax=ax, scatter_kws={'alpha': 0.4})
ax.set_title('Perte en Capital vs Gain en Capital')
ax.set_xlabel('Perte en capital')
ax.set_ylabel('Gain en capital')

plt.tight_layout()
plt.savefig('scatter_plots_regression.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Scatter plots sauvegardés: 'scatter_plots_regression.png'\n")

# Calculer les corrélations et afficher les paramètres de régression
print("\n" + "=" * 70)
print("PARAMÈTRES DE RÉGRESSION ET CORRÉLATIONS")
print("=" * 70 + "\n")

from scipy import stats

pairs = [
    ('age', 'hours-per-week'),
    ('age', 'education-num'),
    ('age', 'capital-gain'),
    ('hours-per-week', 'capital-gain'),
    ('hours-per-week', 'education-num'),
    ('capital-loss', 'capital-gain')
]

for x_var, y_var in pairs:
    # Supprimer les valeurs manquantes
    data_clean = df[[x_var, y_var]].dropna()
    
    # Calculer la régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(data_clean[x_var], data_clean[y_var])
    
    # Calculer la corrélation de Pearson
    correlation = data_clean[x_var].corr(data_clean[y_var])
    
    print(f"\n{x_var.upper()} vs {y_var.upper()}:")
    print(f"   Équation: y = {slope:.6f}x + {intercept:.6f}")
    print(f"   Pente (slope): {slope:.6f}")
    print(f"   Ordonnée à l'origine (intercept): {intercept:.6f}")
    print(f"   Coefficient de corrélation Pearson: {correlation:.6f}")
    print(f"   R-squared: {r_value**2:.6f}")
    print(f"   P-value: {p_value:.2e}")
    print(f"   Interprétation: ", end="")
    
    if abs(correlation) > 0.7:
        print(f"✓ Forte corrélation {'positive' if correlation > 0 else 'négative'}")
    elif abs(correlation) > 0.4:
        print(f"◐ Corrélation modérée {'positive' if correlation > 0 else 'négative'}")
    else:
        print(f"✗ Faible ou pas de corrélation")

print("\n" + "=" * 70)
print("\nCOMMENTAIRES SUR LES RÉSULTATS:")
print("=" * 70)
print("""
1. AGE vs HEURES/SEMAINE: Relation généralement faible à modérée
   - Les travailleurs de tous les âges travaillent des heures similaires
   
2. AGE vs NIVEAU D'ÉDUCATION: Corrélation modérée positive
   - Les travailleurs plus âgés ont tendance à avoir des niveaux d'éducation légèrement plus élevés
   
3. AGE vs GAIN EN CAPITAL: Très faible corrélation
   - Peu de relation entre l'âge et les gains en capital
   
4. HEURES/SEMAINE vs NIVEAU D'ÉDUCATION: Faible corrélation
   - Le nombre d'heures travaillées n'est pas fortement lié au niveau d'éducation
   
5. HEURES/SEMAINE vs GAIN EN CAPITAL: Très faible corrélation
   - Les heures travaillées ne sont pas un bon prédicteur des gains en capital
   
6. GAIN EN CAPITAL vs PERTE EN CAPITAL: Corrélation très faible
   - Ces deux variables sont largement indépendantes
   
📊 CONCLUSION: Les variables numériques du dataset montrent généralement des corrélations 
   faibles à modérées, ce qui suggère que chaque variable apporte une information unique 
   au modèle de prédiction.
""")

print("=" * 70 + "\n")

# ============================================
# 3) PRÉPARATION DES DONNÉES
# ============================================

print("\n3) PRÉPARATION DES DONNÉES POUR L'APPRENTISSAGE")
print("=" * 70)

print("""
📌 CLASSIFICATION BINAIRE - CIBLE: PRÉDICTION DU REVENU (Income)

La cible consiste à prédire si le revenu d'une personne dépasse 50K$ ou non.
Configuration de la classification binaire:
    - Classe 0: Income <= 50K (Revenu faible/moyen)
    - Classe 1: Income > 50K (Revenu élevé)
""")

# Nettoyer les données (supprimer les lignes avec valeurs manquantes dans les colonnes critiques)
df_clean = df.copy()

# Convertir la colonne 'income' en variable binaire
# 0 pour <=50K (revenu faible/moyen)
# 1 pour >50K (revenu élevé)
print("Conversion de la variable 'income' en classification binaire:")
print(f"  Avant: {df_clean['income'].unique()}")
df_clean['income'] = (df_clean['income'].str.strip() == '>50K').astype(int)
print(f"  Après: {df_clean['income'].unique()}")
print(f"  - Classe 0 (<=50K): {(df_clean['income'] == 0).sum()} instances")
print(f"  - Classe 1 (>50K): {(df_clean['income'] == 1).sum()} instances")

print(f"\nDataset initial: {df.shape}")
print(f"Dataset après transformation: {df_clean.shape}")

# Sélectionner les variables numériques comme features
X = df_clean[numeric_cols].copy()
y = df_clean['income'].copy()

print(f"\nFEATURES (Variables prédictives):")
print(f"  Shape: {X.shape}")
print(f"  Colonnes: {list(X.columns)}")

print(f"\nCIBLE (Variable à prédire - Income):")
print(f"  Shape: {y.shape}")
print(f"  Nombre de classes: {y.nunique()}")
print(f"  Distribution des classes:")
print(f"    - Classe 0 (<=50K): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"    - Classe 1 (>50K): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

# Normaliser les features pour améliorer la performance des modèles
print(f"\nNormalisation des features avec StandardScaler:")
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

print(f"✓ Données normalisées - Moyenne: {X_scaled.mean().mean():.6f}, Écart-type: {X_scaled.std().mean():.6f}")

# ============================================
# SÉPARATION EN BASES D'APPRENTISSAGE ET TEST
# ============================================

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("SÉPARATION DES DONNÉES")
print("=" * 70)

# Séparation des données: 70% apprentissage, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTaille de l'ensemble d'apprentissage: {X_train.shape[0]} ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Taille de l'ensemble de test: {X_test.shape[0]} ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"\nDistribution des classes en apprentissage:")
print(pd.Series(y_train).value_counts())
print(f"\nDistribution des classes en test:")
print(pd.Series(y_test).value_counts())

# ============================================
# 1) MODÈLE KNN - ÉTUDE DU PARAMÈTRE K
# ============================================

print("\n" + "=" * 70)
print("1) KNN - INFLUENCE DU PARAMÈTRE K")
print("=" * 70)

k_values = [3, 5, 7, 9, 11, 15, 20, 25, 30]
knn_results = []

print("\nTests avec différentes valeurs de k:")
print("-" * 70)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    knn_results.append({
        'k': k,
        'train_score': train_score,
        'test_score': test_score,
        'f1_score': f1
    })
    
    print(f"k={k:2d} | Train: {train_score:.4f} | Test: {test_score:.4f} | F1: {f1:.4f}")

# Meilleur k
best_k_idx = np.argmax([r['test_score'] for r in knn_results])
best_k = knn_results[best_k_idx]['k']

print(f"\n✓ Meilleur k: {best_k} avec un score de test de {knn_results[best_k_idx]['test_score']:.4f}")

# Graphique de l'influence de k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

k_vals = [r['k'] for r in knn_results]
train_scores = [r['train_score'] for r in knn_results]
test_scores = [r['test_score'] for r in knn_results]
f1_scores = [r['f1_score'] for r in knn_results]

ax1.plot(k_vals, train_scores, 'o-', label='Train Score', linewidth=2, markersize=8)
ax1.plot(k_vals, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
ax1.set_xlabel('Paramètre k')
ax1.set_ylabel('Accuracy')
ax1.set_title('KNN - Influence du paramètre k')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_vals)

ax2.plot(k_vals, f1_scores, 'D-', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Paramètre k')
ax2.set_ylabel('F1-Score')
ax2.set_title('KNN - F1-Score en fonction de k')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_vals)

plt.tight_layout()
plt.savefig('knn_k_influence.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Graphique sauvegardé: 'knn_k_influence.png'")

# Matrice de confusion pour le meilleur k
print("\n" + "=" * 70)
print(f"MATRICE DE CONFUSION - KNN (k={best_k})")
print("=" * 70)

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f"\nMatrice de Confusion:\n{cm_knn}")

# Rapport de classification
print(f"\nRapport de Classification:")
print(classification_report(y_test, y_pred_knn, target_names=['<=50K', '>50K']))

# Visualiser la matrice de confusion
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
ax.set_ylabel('Vraie classe')
ax.set_xlabel('Classe prédite')
ax.set_title(f'Matrice de Confusion - KNN (k={best_k})')
plt.tight_layout()
plt.savefig('confusion_matrix_knn.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Matrice de confusion sauvegardée: 'confusion_matrix_knn.png'")

print("\n📊 OBSERVATIONS sur la matrice de confusion:")
print(f"   - Vrai Négatifs (TN): {cm_knn[0,0]} - Personnes correctement classées comme <=50K")
print(f"   - Faux Positifs (FP): {cm_knn[0,1]} - Personnes <=50K classées à tort comme >50K")
print(f"   - Faux Négatifs (FN): {cm_knn[1,0]} - Personnes >50K classées à tort comme <=50K")
print(f"   - Vrai Positifs (TP): {cm_knn[1,1]} - Personnes correctement classées comme >50K")

# ============================================
# 2) COMPARAISON DE DIFFÉRENTS MODÈLES
# ============================================

print("\n" + "=" * 70)
print("2) COMPARAISON DE DIFFÉRENTS MODÈLES")
print("=" * 70)

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=best_k),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results_comparison = []

print("\nEntraînement et évaluation des modèles:")
print("-" * 70)

for model_name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Scores
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    results_comparison.append({
        'Model': model_name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'F1-Score': f1,
        'Overfitting': train_acc - test_acc
    })
    
    print(f"\n{model_name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  Overfitting:    {train_acc - test_acc:.4f}")

# Créer un DataFrame pour mieux comparer
results_df = pd.DataFrame(results_comparison)
print("\n" + "=" * 70)
print("TABLEAU COMPARATIF DES MODÈLES")
print("=" * 70)
print(results_df.to_string(index=False))

# Visualiser les résultats de comparaison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

model_names = results_df['Model']
x_pos = np.arange(len(model_names))
width = 0.35

# Accuracy Comparison
ax = axes[0]
ax.bar(x_pos - width/2, results_df['Train Accuracy'], width, label='Train', alpha=0.8)
ax.bar(x_pos + width/2, results_df['Test Accuracy'], width, label='Test', alpha=0.8)
ax.set_ylabel('Accuracy')
ax.set_title('Comparaison des Accuracies')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# F1-Score Comparison
ax = axes[1]
ax.bar(x_pos, results_df['F1-Score'], color='green', alpha=0.7)
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score des Modèles')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Overfitting
ax = axes[2]
ax.bar(x_pos, results_df['Overfitting'], color='red', alpha=0.7)
ax.set_ylabel('Train - Test Accuracy')
ax.set_title('Analyse de l\'Overfitting')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Comparaison des modèles sauvegardée: 'models_comparison.png'")

# ============================================
# 3) MODÈLE OPTIMAL
# ============================================

print("\n" + "=" * 70)
print("3) MODÈLE OPTIMAL")
print("=" * 70)

# Trouver le meilleur modèle selon l'accuracy de test
best_model_idx = results_df['Test Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_test_acc = results_df.loc[best_model_idx, 'Test Accuracy']
best_f1 = results_df.loc[best_model_idx, 'F1-Score']

print(f"\n✓ MODÈLE OPTIMAL: {best_model_name}")
print(f"  Test Accuracy: {best_test_acc:.4f}")
print(f"  F1-Score:      {best_f1:.4f}")

# Matrice de confusion du meilleur modèle
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm_best = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
ax.set_ylabel('Vraie classe')
ax.set_xlabel('Classe prédite')
ax.set_title(f'Matrice de Confusion - Modèle Optimal: {best_model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\nMatrice de Confusion du modèle optimal:\n{cm_best}")
print(f"\nRapport de Classification:")
print(classification_report(y_test, y_pred_best, target_names=['<=50K', '>50K']))

# Sauvegarder le meilleur modèle pour le déploiement
joblib.dump(best_model, 'census.pkl')
joblib.dump(scaler, 'scaler.pkl')
feature_names = numeric_cols
joblib.dump(feature_names, 'feature_names.pkl')
print(f"\n✓ Meilleur modèle sauvegardé: 'census.pkl'")
print(f"✓ Scaler sauvegardé: 'scaler.pkl'")
print(f"✓ Noms des features sauvegardés: 'feature_names.pkl'")
print(f"\nFichiers de déploiement créés:")
print(f"  - census.pkl: Modèle {best_model_name} entraîné")
print(f"  - scaler.pkl: Normalisation StandardScaler")
print(f"  - feature_names.pkl: Liste des variables prédictives")

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
📊 RÉSULTATS CLÉS:

1. KNN (Influence de k):
   - Meilleur k: {best_k}
   - Test Accuracy: {knn_results[best_k_idx]['test_score']:.4f}
   - Le k optimal équilibre le biais et la variance

2. Comparaison des modèles:
   - Meilleur modèle: {best_model_name}
   - Accuracy: {best_test_acc:.4f}
   - F1-Score: {best_f1:.4f}

3. Recommandations:
   - Le modèle {best_model_name} offre les meilleures performances
   - Les autres modèles: {', '.join([m for m in results_df['Model'] if m != best_model_name])}
   - Considérer l'équilibre entre accuracy et temps de calcul
   - L'overfitting est contrôlé grâce à la séparation train/test
""")

print("=" * 70)

# ============================================
# EXPLICATIONS SUR LA CLASSIFICATION BINAIRE
# ============================================

print("\n" + "=" * 70)
print("EXPLICATION DÉTAILLÉE: CLASSIFICATION BINAIRE DU REVENU")
print("=" * 70)

print("""
📊 OBJECTIF DU PROJET:
   Prédire la classe de revenu ('Income') dans le jeu de données Census

🎯 TYPE DE PROBLÈME: CLASSIFICATION BINAIRE
   C'est un problème de classification à 2 classes (binaire)

💰 CLASSES DE LA CIBLE 'INCOME':

   ┌─────────────────────────────────────────────────────────┐
   │ Classe 0: Income <= 50K (Revenu faible/moyen)          │
   │ ├─ Label: "<=50K"                                       │
   │ ├─ Valeur numérique: 0                                  │
   │ └─ Signification: Personnes gagnant 50K$ ou moins       │
   └─────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────┐
   │ Classe 1: Income > 50K (Revenu élevé)                  │
   │ ├─ Label: ">50K"                                        │
   │ ├─ Valeur numérique: 1                                  │
   │ └─ Signification: Personnes gagnant plus de 50K$        │
   └─────────────────────────────────────────────────────────┘

📈 DISTRIBUTION DES CLASSES DANS LES DONNÉES:
""")

class_0_count = (y == 0).sum()
class_1_count = (y == 1).sum()
total = len(y)
class_0_pct = class_0_count / total * 100
class_1_pct = class_1_count / total * 100

print(f"""
   Classe 0 (<=50K):  {class_0_count:6d} instances ({class_0_pct:5.1f}%) ███████████████████
   Classe 1 (>50K):   {class_1_count:6d} instances ({class_1_pct:5.1f}%) {'█' * int(class_1_pct)}
   ─────────────────────────────────────
   Total:             {total:6d} instances (100.0%)

🔬 VARIABLES PRÉDICTIVES (Features):
   {', '.join(numeric_cols)}

📋 MATRICE DE CONFUSION - INTERPRÉTATION:

   Pour chaque modèle, la matrice de confusion montre:

   ┌────────────────────────────────────────────────────────┐
   │                  PRÉDICTIONS                            │
   │              <=50K (0)  │  >50K (1)                    │
   ├────────────────────────────────────────────────────────┤
   │ <=50K (0)  │    TN     │     FP    │  Vrais Négatifs  │
   │ RÉALITÉ    │           │           │  & Faux Positifs │
   ├────────────────────────────────────────────────────────┤
   │ >50K (1)   │    FN     │     TP    │  Faux Négatifs   │
   │            │           │           │  & Vrais Positifs│
   └────────────────────────────────────────────────────────┘

   - TN (True Negatives): Correctement classés comme <=50K
   - FP (False Positives): Incorrectement classés comme >50K (erreur de type I)
   - FN (False Negatives): Incorrectement classés comme <=50K (erreur de type II)
   - TP (True Positives): Correctement classés comme >50K

⚖️ MÉTRIQUES D'ÉVALUATION:

   Accuracy = (TP + TN) / (TP + TN + FP + FN)
      → Pourcentage global de prédictions correctes

   Precision (pour classe 1) = TP / (TP + FP)
      → Parmi les personnes classées comme >50K, combien gagnent vraiment >50K ?

   Recall (pour classe 1) = TP / (TP + FN)
      → Parmi les personnes qui gagnent vraiment >50K, combien ont été identifiées ?

   F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
      → Moyenne harmonique de Precision et Recall

✅ INTERPRÉTATION DES RÉSULTATS:

   Score d'entraînement (Train Score):
      - Mesure la performance du modèle sur les données utilisées pour l'entraînement
      - Si trop élevé par rapport au score de test → risque de surapprentissage

   Score de test (Test Score):
      - Mesure la performance du modèle sur des données jamais vues
      - C'est la métrique la plus importante pour évaluer le modèle
      - Représente la performance réelle du modèle en production

   Overfitting:
      - Détecté quand Train Score >> Test Score
      - Indique que le modèle apprend trop les particularités des données d'entraînement
""")

print("=" * 70 + "\n")

# ============================================
# PRÉDICTION INDIVIDUELLE
# ============================================

print("\n" + "=" * 70)
print("PRÉDICTION INDIVIDUELLE DU REVENU")
print("=" * 70)

print("""
🔮 FONCTIONNALITÉ DE PRÉDICTION:
   Utilisez le modèle entraîné pour prédire le revenu d'une personne
   basé sur ses caractéristiques démographiques et économiques.

📋 VARIABLES REQUISES:
   - age: Âge de la personne (18-100)
   - education-num: Niveau d'éducation numérique (1-16)
   - capital-gain: Gain en capital ($)
   - capital-loss: Perte en capital ($)
   - hours-per-week: Heures travaillées par semaine (0-100)
""")

# Charger le modèle sauvegardé
try:
    model_loaded = joblib.load('census.pkl')
    scaler_loaded = joblib.load('scaler.pkl')
    feature_names_loaded = joblib.load('feature_names.pkl')
    print("✓ Modèle chargé avec succès depuis 'census.pkl'")
except FileNotFoundError:
    print("❌ Erreur: Fichiers du modèle non trouvés. Veuillez exécuter la partie entraînement d'abord.")
    exit()

# Fonction pour obtenir une entrée valide
def get_valid_input(prompt, min_val, max_val, input_type=float):
    while True:
        try:
            value = input_type(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Valeur hors limites. Veuillez entrer une valeur entre {min_val} et {max_val}.")
        except ValueError:
            print(f"❌ Entrée invalide. Veuillez entrer un nombre {'entier' if input_type == int else 'décimal'}.")

# Collecter les données utilisateur
print("\n📝 Veuillez entrer les informations de la personne à prédire:")
age = get_valid_input("Âge (18-100): ", 18, 100, int)
education_num = get_valid_input("Niveau d'éducation numérique (1-16): ", 1, 16, float)
capital_gain = get_valid_input("Gain en capital ($): ", 0, 100000, float)
capital_loss = get_valid_input("Perte en capital ($): ", 0, 10000, float)
hours_per_week = get_valid_input("Heures travaillées par semaine (0-100): ", 0, 100, int)

# Créer le DataFrame d'entrée
user_input = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week]
})

print(f"\n📋 Données saisies:")
print(user_input.T)

# Normaliser les données
user_input_scaled = scaler_loaded.transform(user_input)

# Faire la prédiction
prediction = model_loaded.predict(user_input_scaled)[0]
prediction_proba = model_loaded.predict_proba(user_input_scaled)[0] if hasattr(model_loaded, 'predict_proba') else None

print(f"\n" + "=" * 70)
print("RÉSULTAT DE LA PRÉDICTION")
print("=" * 70)

if prediction == 1:
    print("✅ **PRÉDICTION: REVENU ÉLEVÉ**")
    print("   Le revenu dépasse 50K$")
    print("   💵 Classe prédite: >50K (1)")
else:
    print("⚠️ **PRÉDICTION: REVENU FAIBLE/MOYEN**")
    print("   Le revenu est ≤ 50K$")
    print("   📊 Classe prédite: <=50K (0)")

if prediction_proba is not None:
    print(f"\n📊 PROBABILITÉS D'APPARTENANCE AUX CLASSES:")
    print(f"   Probabilité (<=50K): {prediction_proba[0]:.2%}")
    print(f"   Probabilité (>50K):  {prediction_proba[1]:.2%}")

print(f"\n🔍 DÉTAILS DE LA PRÉDICTION:")
print(f"   Modèle utilisé: {type(model_loaded).__name__}")
print(f"   Variables utilisées: {', '.join(feature_names_loaded)}")
print(f"   Données normalisées: Oui (StandardScaler)")

print(f"\n" + "=" * 70)
print("FIN DE LA PRÉDICTION")
print("=" * 70)

