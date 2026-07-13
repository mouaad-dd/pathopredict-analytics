import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df_train = pd.read_csv('Training.csv')
df_test = pd.read_csv('Testing.csv')

if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop(columns=['Unnamed: 133'])

X_train = df_train.drop(columns=['prognosis'])
y_train = df_train['prognosis']

X_test = df_test.drop(columns=['prognosis'])
y_test = df_test['prognosis']

liste_symptomes = list(X_train.columns)
joblib.dump(liste_symptomes, 'liste_symptomes.pkl')

print(f"Nombre de symptômes à analyser : {len(liste_symptomes)}")
print(f"Exemple de symptômes : {liste_symptomes[:5]}")

print("\nEntraînement du modèle en cours...")
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X_train, y_train)
print("Modèle entraîné avec succès !")

predictions = modele.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"\nPrécision du modèle sur les données de test : {score * 100:.2f}%")

joblib.dump(modele, 'modele_prediction_maladies.pkl')
print("\nFichiers sauvegardés : 'modele_prediction_maladies.pkl' et 'liste_symptomes.pkl' sont prêts !")