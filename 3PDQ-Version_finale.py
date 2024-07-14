#!/usr/bin/env python
# coding: utf-8

# # PROJET 3PDQ

# ### Installation des packages 

# In[234]:


# Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


# In[235]:


# Chargez le fichier CSV en spécifiant le délimiteur
dataset = pd.read_csv("PDQ3_Version finale.csv", delimiter=";")
dataset


# In[236]:


dataset.shape


# In[237]:


print(dataset.columns.tolist())


# In[238]:


df= dataset[['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10', 'item11', 'item12', 'item13', 'item14', 'item15', 'item16', 'item17', 'item18', 
'item19', 'item20', 'item21', 'item22', 'item23', 'item24', 'item25', 'item26', 'item27', 'item28','Douleur_Central']]
df


# In[239]:


df.shape


# In[240]:


from IPython.display import Image

# Afficher l'image
Image(filename="C:/Users/aliah/Downloads/Capture d’écran 2023-04-21 140544.png")


# ### Vérifier le type de chaque variable 

# In[241]:


df.select_dtypes(object).columns


# ### Conversion de toutes les variables en float

# In[242]:


# Parcours de chaque colonne du DataFrame
for column in df.columns:
    # Conversion de la colonne en type float en ignorant les erreurs de conversion
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Afficher les types de données des colonnes
print(df.dtypes)


# In[243]:


# Parcours de chaque colonne du DataFrame
for column in df.columns:
    # Conversion de la colonne en type float
    df[column] = df[column].astype(float)

# Afficher les types de données des colonnes
print(df.dtypes)


# # Gestion des données manquantes

# In[244]:


#Compter le nombre de valeurs manquantes par variable
missing_values_count = df.isnull().sum()

# Afficher le nombre de valeurs manquantes par variable
print("Nombre de valeurs manquantes par variable :\n", missing_values_count)

# Afficher les variables qui contiennent des valeurs manquantes
missing_variables = df.columns[df.isnull().any()].tolist()
print("\nVariables avec des valeurs manquantes :", missing_variables)


# In[245]:


df.isnull().sum().sum()


# In[246]:


# Calculer le nombre total de cellules dans le DataFrame
total_cells = df.shape[0] * df.shape[1]
total_cells


# In[247]:


# Calculer le nombre total de valeurs manquantes dans le DataFrame
total_missing = df.isnull().sum().sum()

# Calculer la proportion totale de valeurs manquantes dans le DataFrame
proportion_missing = total_missing / total_cells * 100

# Afficher la proportion totale de valeurs manquantes en pourcentage
print('Proportion totale de valeurs manquantes : {:.2f}%'.format(proportion_missing))


# In[248]:


proportion_missing /100


# In[249]:


# Suppression des données manquantes 
df = df.dropna()
#Vérifier à nouveau
df.isnull().sum().sum()


# In[250]:


df.shape


# In[251]:


df


# ### HEATMAP

# In[252]:


# Calculer la matrice de corrélation
corr_matrix = df.corr()

# Plot du heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de corrélation pour les variables binaires')
plt.show()


# # MODELISATION

# In[253]:


# 1) Créer une matrice des variables indépendantes et le vecteur de la variable dépendante.
# X est la matrice et Y est le vecteur
# La matrice des variables indépendantes est aussi appeelée matrice de featuresµ

X = df.drop('Douleur_Central', axis=1)  # Supprimer la colonne "target" de la matrice X
Y = df['Douleur_Central']              # Sélectionner uniquement la colonne "target" pour Y


# In[254]:


X


# In[255]:


Y


# ### Séparation du dataset en training_set et en test_set

# In[256]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state= 0)

# Calculer la proportion de chaque ensemble
train_prop = len(X_train) / len(X)
test_prop = len(X_test) / len(X)

print("Proportion de données dans l'ensemble d'entraînement: {:.2f}".format(train_prop))
print("Proportion de données dans l'ensemble de test: {:.2f}".format(test_prop))


# # 1- Regression Logistique 

# Les metrics calculés que j'ai calculé pour chaque modèle de machine learning. L'outil diagnostique c'est le total  des six items que j'ai calculé après avoir avoir fait le rfe. et on veut maintenant le seuil de score ou la personne quitte l'état d'un malade non atteint de la douleur parkinson à un malade atteint de la douleur de parkinson. 
# 
# donner à chaque item si oui 1 si non 0 et le score de l'outil diagnostique est le total des réponses 

# In[257]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Créer une instance du modèle de régression logistique
RL = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

# Créer une instance de RFE avec le modèle de régression logistique
rfe = RFE(estimator=RL, n_features_to_select=6)

# Adapter RFE sur les données d'entraînement
rfe.fit(X_train, Y_train)

# Obtenir les indices des fonctionnalités sélectionnées
selected_features_indices = rfe.get_support(indices=True)

# Obtenir les noms des fonctionnalités sélectionnées
selected_features_names = X_train.columns[selected_features_indices]

# Entraîner le modèle de régression logistique sur les fonctionnalités sélectionnées
RL.fit(X_train[selected_features_names], Y_train)

# Afficher les variables du feature selection
print("Variables du feature selection : ")
print(selected_features_names)


# In[258]:


# Prédire les classes pour les données de test
Y_pred1 = rfe.predict(X_test)
Y_pred1


# In[259]:


from sklearn.metrics import confusion_matrix
CM1 = confusion_matrix(Y_test, Y_pred1)
CM1


# In[260]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM1.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM1.flatten()/np.sum(CM1)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM1, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix Regression Logistique', fontsize=16)
plt.tight_layout()
plt.show()


# In[261]:


from sklearn import metrics

y_pred_proba1 = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba1)
auc = metrics.roc_auc_score(Y_test, y_pred_proba1)

plt.plot(fpr, tpr, label="Régression Logistique, AUC = {:.2f}".format(auc))
plt.legend(loc="lower right")
plt.title('Courbe ROC de la Régression Logistique', fontsize=16)
plt.xlabel('Taux de faux positifs', fontsize=12)
plt.ylabel('Taux de vrais positifs', fontsize=12)
plt.grid(True)
plt.show()


# In[262]:


# Calculer les taux de faux positifs et de vrais positifs
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba1)

# Calculer les scores en utilisant les taux de vrais positifs et les taux de faux positifs
scores = tpr - fpr

# Trouver l'indice du score maximum
best_index = np.argmax(scores)

# Obtenir le seuil de classification correspondant au score maximum
best_threshold = thresholds[best_index]

# Afficher les scores individuellement
for score, threshold in zip(scores, thresholds):
    print("Score : {:.2f}, Seuil : {:.2f}".format(score, threshold))

# Afficher le score maximum et le seuil correspondant
print("Meilleur score : {:.2f}".format(scores[best_index]))
print("Seuil correspondant : {:.2f}".format(best_threshold))


# In[263]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Calculer les métriques
Accuracy_Rate1 = accuracy_score(Y_test, Y_pred1)
F1_score1 = f1_score(Y_test, Y_pred1)
Precision1 = precision_score(Y_test, Y_pred1)
Recall1 = recall_score(Y_test, Y_pred1)
CK1 = cohen_kappa_score(Y_test, Y_pred1)
MC1 = matthews_corrcoef(Y_test, Y_pred1)
auc1 = metrics.roc_auc_score(Y_test, y_pred_proba1)

# Calculer les intervalles de confiance
conf_level = 0.95

# Create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "F1 Score", "Cohen's Kappa", "Matthews Corrcoef", "AUC"]
metric_values = [Precision1, Recall1, Accuracy_Rate1, F1_score1, CK1, MC1, auc1]

# Calculate confidence intervals using normal distribution approximation
conf_intervals = [stats.norm.interval(conf_level, loc=metric_val, scale=stats.sem(Y_pred1 == Y_test)) for metric_val in metric_values]

# Print metric names, values, and confidence intervals
for name, value, interval in zip(metric_names, metric_values, conf_intervals):
    print(f"{name}: {value:.2f} (CI: {interval[0]:.2f} - {interval[1]:.2f})")

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(metric_names, metric_values, yerr=np.array(conf_intervals).T)
ax.set_ylabel('Value')
ax.set_title('Performance Metrics')
plt.show()


# # 2- Random Forest 

# In[264]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# Convertir X_train en tableau NumPy
X_train_np = X_train.values

# Créer une instance du modèle Random Forest
RF = RandomForestClassifier(n_estimators=100000, n_jobs=1, max_depth=2,random_state = 0)

# Créer une instance de Boruta
feat_selector = BorutaPy(RF, n_estimators='auto', max_iter=100, verbose=2, random_state = 0)

# Adapter Boruta sur les données d'entraînement
feat_selector.fit(X_train_np, Y_train)

# Obtenir les indices des fonctionnalités sélectionnées
selected_features_indices = feat_selector.support_

# Sélectionner les fonctionnalités avec Boruta
selected_features = X_train.columns[feat_selector.support_]

# Adapter le modèle Random Forest sur les fonctionnalités sélectionnées
RF.fit(X_train[selected_features], Y_train)

# Afficher les variables du feature selection
print("Variables du feature selection : ")
print(selected_features_names)


# In[265]:


# Prédire les classes pour les données de test en utilisant les mêmes fonctionnalités sélectionnées
Y_pred2 = RF.predict(X_test[selected_features])

# Afficher les prédictions
print(Y_pred2)


# In[266]:


CM2 = confusion_matrix(Y_test, Y_pred2)
CM2


# In[267]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM2.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM2.flatten()/np.sum(CM2)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM2, annot=labels, fmt='', cmap='Pastel1')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix RANDOM FOREST', fontsize=16)
plt.tight_layout()
plt.show()


# In[268]:


y_pred_proba2 = RF.predict_proba(X_test[selected_features])[:, 1]

# Calculer les taux de faux positifs et de vrais positifs
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba2)

auc2 = metrics.roc_auc_score(Y_test, y_pred_proba2)

plt.plot(fpr, tpr, label="Random Forest, AUC = {:.2f}".format(auc2))
plt.legend(loc="lower right")
plt.title('Courbe ROC du Random Forest', fontsize=16)
plt.xlabel('Taux de faux positifs', fontsize=12)
plt.ylabel('Taux de vrais positifs', fontsize=12)
plt.grid(True)
plt.show()


# In[269]:


# Calculer le score pour y_pred_proba2
score = metrics.roc_auc_score(Y_test, y_pred_proba2)

# Afficher le score
print("Score pour y_pred_proba2 : {:.2f}".format(score))


# Pour calculer le score, la fonction roc_auc_score compare les probabilités prédites (y_pred_proba2) avec les vraies étiquettes de classe (Y_test). Elle calcule ensuite l'aire sous la courbe ROC en intégrant le taux de vrais positifs par rapport au taux de faux positifs sur toute la plage des seuils de classification.
# 
# Ce score est une mesure de la capacité du modèle à classer correctement les instances des deux classes, en prenant en compte à la fois la sensibilité (capacité à détecter les vrais positifs) et la spécificité (capacité à éviter les faux positifs).

# ### Présenter toujours la courbe roc avec l'indice du score maximum et le seuil de classification et les couples de spécificité et de sensiblité 

# In[270]:


# Calculer les taux de faux positifs et de vrais positifs
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba2)

# Calculer les scores en utilisant les taux de vrais positifs et les taux de faux positifs
scores = tpr - fpr

# Trouver l'indice du score maximum
best_index = np.argmax(scores)

# Obtenir le seuil de classification correspondant au score maximum
best_threshold = thresholds[best_index]

# Afficher les scores individuellement
for score, threshold in zip(scores, thresholds):
    print("Score : {:.2f}, Seuil : {:.2f}".format(score, threshold))

# Afficher le score maximum et le seuil correspondant
print("Meilleur score : {:.2f}".format(scores[best_index]))
print("Seuil correspondant : {:.2f}".format(best_threshold))


# In[271]:


# Calculer les taux de faux positifs et de vrais positifs
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba2)

# Calculer la spécificité
specificity = 1 - fpr

# Afficher les couples de spécificité et de sensibilité
for spec, sens in zip(specificity, tpr):
    print("Spécificité: {:.2f}, Sensibilité: {:.2f}".format(spec, sens))

# Tracer la courbe ROC
plt.plot(fpr, tpr)
plt.xlabel('Taux de faux positifs (1 - Spécificité)')
plt.ylabel('Taux de vrais positifs (Sensibilité)')
plt.title('Courbe ROC')
plt.grid(True)
plt.show()

# Calculer l'aire sous la courbe ROC (AUC)
auc = metrics.roc_auc_score(Y_test, y_pred_proba2)

# Afficher la valeur de l'aire sous la courbe ROC
print("Aire sous la courbe ROC (AUC) : {:.2f}".format(auc))


# ### Intervalle de confiance pour la courbe roc 

# In[272]:


import numpy as np
from sklearn import metrics

# Réinitialiser l'index de Y_test
Y_test = Y_test.reset_index(drop=True)

# Calculer l'aire sous la courbe ROC (AUC) pour les prédictions
auc = metrics.roc_auc_score(Y_test, y_pred_proba2)

# Nombre d'échantillons bootstrap
n_bootstrap = 1000

# Liste pour stocker les valeurs d'AUC bootstrap
auc_bootstrap = []

# Effectuer le bootstrap
for _ in range(n_bootstrap):
    # Générer un échantillon bootstrap en rééchantillonnant les prédictions avec remplacement
    indices = np.random.choice(len(y_pred_proba2), len(y_pred_proba2), replace=True)
    bootstrap_preds = y_pred_proba2[indices]
    bootstrap_labels = Y_test[indices]
    
    # Réinitialiser l'index de bootstrap_labels
    bootstrap_labels = bootstrap_labels.reset_index(drop=True)
    
    # Calculer l'AUC pour l'échantillon bootstrap
    bootstrap_auc = metrics.roc_auc_score(bootstrap_labels, bootstrap_preds)
    
    # Ajouter l'AUC bootstrap à la liste
    auc_bootstrap.append(bootstrap_auc)

# Calculer l'intervalle de confiance à 95% pour l'AUC
lower = np.percentile(auc_bootstrap, 2.5)
upper = np.percentile(auc_bootstrap, 97.5)

# Afficher l'AUC et l'intervalle de confiance
print("AUC:", auc)
print("Intervalle de confiance à 95%:", (lower, upper))


# In[273]:


# Définir le seuil
seuil = 0.5

# Calculer les vrais positifs, les faux positifs, les vrais négatifs et les faux négatifs
vrais_positifs = np.sum((Y_test == 1) & (y_pred_proba2 >= seuil))
faux_negatifs = np.sum((Y_test == 1) & (y_pred_proba2 < seuil))
vrais_negatifs = np.sum((Y_test == 0) & (y_pred_proba2 < seuil))
faux_positifs = np.sum((Y_test == 0) & (y_pred_proba2 >= seuil))

# Calculer la sensibilité (rappel)
sensibilite = vrais_positifs / (vrais_positifs + faux_negatifs)

# Calculer la spécificité
specificite = vrais_negatifs / (vrais_negatifs + faux_positifs)

# Afficher les valeurs de sensibilité et de spécificité
print("Sensibilité (Rappel) : {:.2f}".format(sensibilite))
print("Spécificité : {:.2f}".format(specificite))


# In[274]:


# Définir le seuil
seuil = 0.31

# Calculer les vrais positifs, les faux positifs, les vrais négatifs et les faux négatifs
vrais_positifs = np.sum((Y_test == 1) & (y_pred_proba2 >= seuil))
faux_negatifs = np.sum((Y_test == 1) & (y_pred_proba2 < seuil))
vrais_negatifs = np.sum((Y_test == 0) & (y_pred_proba2 < seuil))
faux_positifs = np.sum((Y_test == 0) & (y_pred_proba2 >= seuil))

# Calculer la sensibilité (rappel)
sensibilite = vrais_positifs / (vrais_positifs + faux_negatifs)

# Calculer la spécificité
specificite = vrais_negatifs / (vrais_negatifs + faux_positifs)

# Afficher les valeurs de sensibilité et de spécificité
print("Sensibilité (Rappel) : {:.2f}".format(sensibilite))
print("Spécificité : {:.2f}".format(specificite))


# In[275]:


Accuracy_Rate2 = accuracy_score(Y_test, Y_pred2)
Error_rate2 = 1 - Accuracy_Rate2
F1_score2 = f1_score(Y_test, Y_pred2)
Precision2 = precision_score(Y_test, Y_pred2)
Recall2 = recall_score(Y_test, Y_pred2)
CK2 = cohen_kappa_score (Y_test,Y_pred2)
MC2 = matthews_corrcoef(Y_test,Y_pred2)
auc2 = metrics.roc_auc_score(Y_test, y_pred_proba2)

print("Precision : {:.2f}".format(Precision2))
print("Recall OU SENSIBILITE: {:.2f}".format(Recall2))
print("Accuracy rate: ", Accuracy_Rate2)
print("Error rate : ",Error_rate2)
print("F1_score: ", F1_score2)
print("CK:", CK2)
print("MC:", MC2)
print("AUC:", auc2)


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [Precision2, Recall2, Accuracy_Rate2, Error_rate2, F1_score2, CK2, MC2,auc2]

# create a bar chart
fig, ax = plt.subplots(figsize=(10,10))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# In[276]:


import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score

# Définir les métriques à évaluer
metrics = {
    "Accuracy Rate": accuracy_score,
    "Error Rate": lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred),
    "F1 Score": f1_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "Cohen's Kappa": cohen_kappa_score,
    "Matthews Correlation Coefficient": matthews_corrcoef,
    "AUC": roc_auc_score
}

# Nombre d'échantillons bootstrap
n_iterations = 1000

# Initialiser un dictionnaire pour stocker les intervalles de confiance
confidence_intervals = {}

# Boucle à travers les métriques
for metric_name, metric_func in metrics.items():
    # Liste pour stocker les métriques échantillonnées
    metric_samples = []

    # Boucle pour effectuer le bootstrap
    for _ in range(n_iterations):
        # Échantillonnage bootstrap des prédictions
        indices = np.random.choice(len(y_pred_proba2), len(y_pred_proba2), replace=True)
        bootstrap_preds = y_pred_proba2[indices]
        bootstrap_labels = Y_test[indices]

        # Calcul de la métrique pour l'échantillon bootstrap
        metric_value = metric_func(bootstrap_labels, np.round(bootstrap_preds))
        metric_samples.append(metric_value)

    # Calcul de l'intervalle de confiance à 95% pour la métrique
    confidence_interval = np.percentile(metric_samples, [2.5, 97.5])
    confidence_intervals[metric_name] = confidence_interval

    print(f"{metric_name} Confidence Interval (95%):", confidence_interval)


# # 3- Gradient Boosting

# In[277]:


from sklearn.ensemble import GradientBoostingClassifier


# In[280]:


# Créer une instance du modèle Gradient Boosting
GB = GradientBoostingClassifier(n_estimators=100000, max_depth=3, min_samples_leaf=1, learning_rate=0.1, random_state=0)

# Adapter le modèle Gradient Boosting sur les données d'entraînement
GB.fit(X_train, Y_train)

# Prédire les classes pour les données de test
Y_pred = GB.predict(X_test)


# In[281]:


CM3 = confusion_matrix(Y_test, Y_pred)
CM3


# In[282]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM3.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM3.flatten()/np.sum(CM3)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM3, annot=labels, fmt='', cmap='crest')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix GRADIENT BOOSTING', fontsize=16)
plt.tight_layout()
plt.show()


# In[284]:


from sklearn.metrics import roc_curve

y_pred_proba3 = GB.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba3)
auc3 = metrics.roc_auc_score(Y_test, y_pred_proba3)

plt.plot(fpr, tpr, label="Gradient Boosting, AUC = {:.2f}".format(auc3))
plt.legend(loc="lower right")
plt.title('Courbe ROC du Gradient Boosting', fontsize=16)
plt.xlabel('Taux de faux positifs', fontsize=12)
plt.ylabel('Taux de vrais positifs', fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:


# Calculer les taux de faux positifs et de vrais positifs
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba3)

# Calculer les scores en utilisant les taux de vrais positifs et les taux de faux positifs
scores = tpr - fpr

# Trouver l'indice du score maximum
best_index = np.argmax(scores)

# Obtenir le seuil de classification correspondant au score maximum
best_threshold = thresholds[best_index]

# Afficher les scores individuellement
for score, threshold in zip(scores, thresholds):
    print("Score : {:.2f}, Seuil : {:.2f}".format(score, threshold))

# Afficher le score maximum et le seuil correspondant
print("Meilleur score : {:.2f}".format(scores[best_index]))
print("Seuil correspondant : {:.2f}".format(best_threshold))


# In[ ]:


Accuracy_Rate3 = accuracy_score(Y_test, Y_pred)
Error_rate3 = 1 - Accuracy_Rate3
F1_score3 = f1_score(Y_test, Y_pred)
Precision3 = precision_score(Y_test, Y_pred)
Recall3 = recall_score(Y_test, Y_pred)
CK3 = cohen_kappa_score (Y_test,Y_pred)
MC3 = matthews_corrcoef(Y_test,Y_pred)
auc3 = metrics.roc_auc_score(Y_test, y_pred_proba3)

print("Precision : {:.2f}".format(Precision3))
print("Recall : {:.2f}".format(Recall3))
print("Accuracy rate: ", Accuracy_Rate3)
print("Error rate: ",Error_rate3)
print("F1_score: ", F1_score3)
print("CK:", CK3)
print("MC:", MC3)
print("AUC:", auc3)
# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [Precision3, Recall3, Accuracy_Rate3, Error_rate3, F1_score3, CK3, MC3,auc3]

# create a bar chart
fig, ax = plt.subplots(figsize=(10,10))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# In[ ]:


import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score

# Définir les métriques à évaluer
metrics = {
    "Accuracy Rate": accuracy_score,
    "Error Rate": lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred),
    "F1 Score": f1_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "Cohen's Kappa": cohen_kappa_score,
    "Matthews Correlation Coefficient": matthews_corrcoef,
    "AUC": roc_auc_score
}

# Nombre d'échantillons bootstrap
n_iterations = 1000

# Initialiser un dictionnaire pour stocker les intervalles de confiance
confidence_intervals = {}

# Boucle à travers les métriques
for metric_name, metric_func in metrics.items():
    # Liste pour stocker les métriques échantillonnées
    metric_samples = []

    # Boucle pour effectuer le bootstrap
    for _ in range(n_iterations):
        # Échantillonnage bootstrap des prédictions
        indices = np.random.choice(len(Y_pred), len(Y_pred), replace=True)
        bootstrap_preds = Y_pred[indices]
        bootstrap_labels = Y_test[indices]

        # Calcul de la métrique pour l'échantillon bootstrap
        metric_value = metric_func(bootstrap_labels, bootstrap_preds)
        metric_samples.append(metric_value)

    # Calcul de l'intervalle de confiance à 95% pour la métrique
    confidence_interval = np.percentile(metric_samples, [2.5, 97.5])
    confidence_intervals[metric_name] = confidence_interval

    print(f"{metric_name} Confidence Interval (95%):", confidence_interval)


# # Comparaison des courbes roc

# In[ ]:


# Calculer les taux de faux positifs et les taux de vrais positifs pour chaque modèle
fpr_rl, tpr_rl, _ = metrics.roc_curve(Y_test, y_pred_proba1)
fpr_rf, tpr_rf, _ = metrics.roc_curve(Y_test, y_pred_proba2)
fpr_gb, tpr_gb, _ = metrics.roc_curve(Y_test, y_pred_proba3)

# Calculer les aires sous la courbe (AUC) pour chaque modèle
auc_rl = metrics.roc_auc_score(Y_test, y_pred_proba1)
auc_rf = metrics.roc_auc_score(Y_test, y_pred_proba2)
auc_gb = metrics.roc_auc_score(Y_test, y_pred_proba3)

# Tracer les courbes ROC pour chaque modèle
plt.plot(fpr_rl, tpr_rl, label="Régression Logistique, AUC = {:.2f}".format(auc_rl))
plt.plot(fpr_rf, tpr_rf, label="Random Forest, AUC = {:.2f}".format(auc_rf))
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting, AUC = {:.2f}".format(auc_gb))

# Afficher la légende et les titres
plt.legend(loc="lower right")
plt.title('Comparaison des courbes ROC', fontsize=16)
plt.xlabel('Taux de faux positifs', fontsize=12)
plt.ylabel('Taux de vrais positifs', fontsize=12)
plt.grid(True)

# Afficher le graphique
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score

# Afficher les métriques
print("Métriques pour Régression Logistique :")
print("Accuracy rate: {:.2f}".format(Accuracy_Rate1))
print("Error rate: {:.2f}".format(Error_rate1))
print("Precision: {:.2f}".format(Precision1))
print("Recall: {:.2f}".format(Recall1))
print("F1 Score: {:.2f}".format(F1_score1))
print("CK: {:.2f}".format(CK1))
print("MC: {:.2f}".format(MC1))
print("AUC: {:.2f}".format(auc1))
print()

print("Métriques pour Random Forest :")
print("Accuracy rate: {:.2f}".format(Accuracy_Rate2))
print("Error rate: {:.2f}".format(Error_rate2))
print("Precision: {:.2f}".format(Precision2))
print("Recall: {:.2f}".format(Recall2))
print("F1 Score: {:.2f}".format(F1_score2))
print("CK: {:.2f}".format(CK2))
print("MC: {:.2f}".format(MC2))
print("AUC: {:.2f}".format(auc2))
print()

print("Métriques pour Gradient Boosting :")
print("Accuracy rate: {:.2f}".format(Accuracy_Rate3))
print("Error rate: {:.2f}".format(Error_rate3))
print("Precision: {:.2f}".format(Precision3))
print("Recall: {:.2f}".format(Recall3))
print("F1 Score: {:.2f}".format(F1_score3))
print("CK: {:.2f}".format(CK3))
print("MC: {:.2f}".format(MC3))
print("AUC: {:.2f}".format(auc3))
print()


# In[ ]:


# Calculer les métriques pour chaque modèle

metrics_rl = {
    'Model': 'Régression Logistique',
    'Accuracy': Accuracy_Rate1,
    'Precision': Precision1,
    'Recall':Recall1,
    'F1 Score':F1_score1,
    'CK':CK1,
    'MC': MC1,
    'AUC':auc1,
    'Error Rate':Error_rate1
}

metrics_rf = {
    'Model': 'Random Forest',
   'Accuracy': Accuracy_Rate2,
    'Precision': Precision2,
    'Recall':Recall2,
    'F1 Score':F1_score2,
    'CK':CK2,
    'MC': MC2,
    'AUC':auc2,
    'Error Rate':Error_rate2
}

metrics_gb = {
    'Model': 'Gradient Boosting',
    'Accuracy': Accuracy_Rate3,
    'Precision': Precision3,
    'Recall':Recall3,
    'F1 Score':F1_score3,
    'CK':CK3,
    'MC': MC3,
    'AUC':auc3,
    'Error Rate':Error_rate3
}

# Créer un dataframe avec les métriques
metrics_df = pd.DataFrame([metrics_rl, metrics_rf, metrics_gb])

# Ploter les métriques
plt.figure(figsize=(18, 10))

# Accuracy
plt.subplot(2, 4, 1)
sns.barplot(x='Model', y='Accuracy', data=metrics_df)
plt.title('Accuracy')
plt.ylim(0, 1)

# Precision
plt.subplot(2, 4, 2)
sns.barplot(x='Model', y='Precision', data=metrics_df)
plt.title('Precision')
plt.ylim(0, 1)

# Recall
plt.subplot(2, 4, 3)
sns.barplot(x='Model', y='Recall', data=metrics_df)
plt.title('Recall')
plt.ylim(0, 1)

# F1 Score
plt.subplot(2, 4, 4)
sns.barplot(x='Model', y='F1 Score', data=metrics_df)
plt.title('F1 Score')
plt.ylim(0, 1)

# Cohen Kappa
plt.subplot(2, 4, 5)
sns.barplot(x='Model', y='CK', data=metrics_df)
plt.title('Cohen Kappa')

# Matthews Corrcoef
plt.subplot(2, 4, 6)
sns.barplot(x='Model', y='MC', data=metrics_df)
plt.title('Matthews Corrcoef')

# AUC
plt.subplot(2, 4, 7)
sns.barplot(x='Model', y='AUC', data=metrics_df)
plt.title('AUC')

# Error Rate
plt.subplot(2, 4, 8)
sns.barplot(x='Model', y='Error Rate', data=metrics_df)
plt.title('Error Rate')

plt.tight_layout()
plt.show()


# ### RANDOM FOREST SCORING 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix

# Adapter le modèle Random Forest
RF.fit(X_train[selected_features], Y_train)

# Prédire les probabilités des classes positives (classe 1) pour les six variables
predicted_probabilities = RF.predict_proba(X_test[selected_features])[:, 1]

# Calculer le score de l'outil diagnostique (somme des probabilités)
diagnostic_score = np.sum(predicted_probabilities)

# Définir le seuil de diagnostic
seuil_diagnostic = 25/100

# Appliquer le seuil pour déterminer si la personne est atteinte de la douleur de Parkinson
diagnostic_result = 1 if diagnostic_score >= seuil_diagnostic else 0

# Comparaison avec les étiquettes réelles et évaluation des performances
recall = recall_score(Y_test, [diagnostic_result] * len(Y_test))
roc_auc = roc_auc_score(Y_test, predicted_probabilities)

# Calcul de la matrice de confusion pour calculer la sensibilité et la spécificité
confusion = confusion_matrix(Y_test, [diagnostic_result] * len(Y_test))
tn, fp, fn, tp = confusion.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Affichage des résultats
print("Diagnostic Score:", diagnostic_score)
print("Diagnostic Result:", diagnostic_result)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("ROC AUC:", roc_auc)


# Examinons de plus près chaque métrique et ce qu'elle signifie dans le contexte de la prédiction de la douleur de Parkinson à l'aide du modèle.
# 
# 1. **Diagnostic Score**: Le score de l'outil diagnostique est la somme des probabilités prédites pour les échantillons de test. Dans ce cas, il est d'environ 20.62. Cela signifie que la somme des probabilités attribuées aux échantillons de test est assez élevée, indiquant que le modèle attribue des probabilités significatives aux échantillons considérés positifs.
# 
# 2. **Diagnostic Result**: Le résultat du diagnostic est 1. Cela signifie que le modèle a estimé que le score de l'outil diagnostique est supérieur au seuil de diagnostic (dans ce cas, 25%), ce qui conduit à diagnostiquer la personne comme atteinte de la douleur de Parkinson. 
# 
# 3. **Recall (Sensibilité)**: La sensibilité est de 1.0. Cela indique que le modèle a correctement identifié tous les cas réels de douleur de Parkinson parmi les échantillons de test positifs. En d'autres termes, il n'y a pas de faux négatifs. Tous les patients atteints de Parkinson ont été correctement diagnostiqués comme tels.
# 
# 4. **Specificity (Spécificité)**: La spécificité est de 0.0. Cela signifie que le modèle n'a pas pu identifier correctement les échantillons négatifs (non atteints de Parkinson). Il y a une absence de vrais négatifs. Cela pourrait être dû à une tendance du modèle à prédire positif pour la plupart des échantillons.
# 
# 5. **ROC AUC**: L'aire sous la courbe ROC (ROC AUC) est de 0.71. Cela mesure la capacité du modèle à distinguer entre les échantillons positifs et négatifs en fonction de leurs probabilités prédites. Un score de 0.71 indique que le modèle a une performance modérée dans la séparation des classes, bien qu'il existe un certain chevauchement entre les deux.
# 
# Globalement, bien que le modèle puisse correctement détecter les cas positifs (personnes atteintes de Parkinson) avec une sensibilité parfaite, il a du mal à distinguer les échantillons négatifs. Cela peut être dû à un déséquilibre entre les classes ou à une certaine imprécision dans la prédiction des échantillons négatifs. Il serait important de revoir les caractéristiques, le modèle et les hyperparamètres pour améliorer la performance de la spécificité, tout en maintenant une sensibilité élevée.

# In[ ]:


# Calculer le score de diagnostic pour chaque patient en fonction des résultats des six items
diagnostic_scores_per_patient = np.sum(X[selected_features], axis=1)

# Afficher les scores de diagnostic individuels
for i, score in enumerate(diagnostic_scores_per_patient):
    print(f"Patient {i + 1}: Diagnostic Score = {score}")

