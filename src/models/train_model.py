import pandas as pd
import numpy as np
from sklearn import ensemble
import joblib

print(joblib.__version__)

# Charger les données
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Identifier les colonnes non numériques
non_numeric_columns = X_train.select_dtypes(exclude=['number']).columns
print("Colonnes non numériques :", non_numeric_columns)

# Nettoyer les valeurs mal encodées
X_train = X_train.applymap(lambda x: str(x).replace("Â\xa0", "").strip() if isinstance(x, str) else x)
X_test = X_test.applymap(lambda x: str(x).replace("Â\xa0", "").strip() if isinstance(x, str) else x)

# Convertir les colonnes en numérique
for col in non_numeric_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Remplacer les valeurs NaN par 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Vérifier les types
print(X_train.dtypes)

# Entraîner le modèle
rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Sauvegarder le modèle
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained with  success.")