# 📘 Explication Détaillée du Notebook LSTM (Claud-IA)

## 🎯 Objectif du Projet
Ce notebook montre pas à pas la mise en place d'un réseau de neurones **LSTM (Long Short-Term Memory)** pour l'analyse de sentiments sur le jeu de données **IMDb**.

---

## 🔧 1. Environnement et Versions
```python
import sys, tensorflow as tf, numpy as np, matplotlib
print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
```
Vérifie la cohérence de l'environnement d'exécution.

---

## 📦 2. Imports et Paramètres
Les bibliothèques principales :
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```
Ces modules servent à charger les données, préparer les séquences, construire le modèle et visualiser les performances.

---

## 🗂️ 3. Chargement et Préparation du Dataset
```python
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train = pad_sequences(X_train, maxlen=200, padding='pre', truncating='pre')
X_test  = pad_sequences(X_test,  maxlen=200, padding='pre', truncating='pre')
```
Chaque critique devient une séquence de 200 indices numériques représentant les mots les plus fréquents.

---

## 🧠 4. Construction du Modèle LSTM
```python
model = Sequential([
    Embedding(10000, 128, name="embedding"),
    LSTM(128, dropout=0.2, name="lstm"),
    Dense(1, activation="sigmoid", name="output")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```
- **Embedding** : transforme les indices de mots en vecteurs denses.
- **LSTM** : apprend les dépendances temporelles entre les mots.
- **Dense + Sigmoïde** : sortie entre 0 et 1 (probabilité d’avis positif).

> ℹ️ L’argument `input_length` est facultatif — Keras l’infère automatiquement.

---

## 🏃 5. Entraînement et Graphiques
```python
es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, callbacks=[es])
```
- **EarlyStopping** : arrête automatiquement l’entraînement avant le surapprentissage.

### 📉 Graphique de la perte
Visualise la diminution de l’erreur pendant l’entraînement.

### 📈 Graphique de l’accuracy
Montre l’évolution du taux de réussite sur l’ensemble d’entraînement et de validation.

---

## ✅ 6. Évaluation
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
```
Évalue le modèle sur des données jamais vues — typiquement **85 % de précision**.

---

## 🧪 7. Prédictions Personnalisées
```python
def predict_text(model, raw_text):
    tokens = [2 if word not in word_index else word_index[word]+3 for word in raw_text.lower().split()]
    seq = pad_sequences([tokens], maxlen=200)
    p = model.predict(seq, verbose=0)[0,0]
    return "Positif" if p >= 0.5 else "Négatif", p
```
Permet de tester le modèle avec ses propres phrases.

---

## 🧭 8. Conclusion
Le modèle LSTM est capable de comprendre le **sentiment général** d’un texte à partir de sa structure séquentielle.  
C’est une architecture clé du deep learning pour le traitement du langage naturel.

---

### 💬 Résumé Oral
> « Le LSTM apprend à mémoriser les informations utiles d’un texte tout en oubliant le reste.  
> Grâce à son mécanisme de portes, il parvient à comprendre les émotions exprimées dans les phrases. »

---

## 🔍 Pour Aller Plus Loin — Améliorations du Modèle LSTM

### 🧭 1. Bidirectional(LSTM)
- **Principe :** lit la séquence dans les deux sens (gauche → droite et droite → gauche).  
- **Avantage :** comprend le contexte complet autour de chaque mot.  
- **Code :**
```python
from tensorflow.keras.layers import Bidirectional, LSTM
model.add(Bidirectional(LSTM(128, dropout=0.2)))
```
- **Effet :** +2 à +5 % d’accuracy sur les tâches textuelles.

### 🧠 2. Embeddings pré-entraînés (GloVe, Word2Vec)
- **Principe :** utiliser des vecteurs de mots déjà appris sur de grands corpus (Wikipedia, Google News…).  
- **Avantage :** meilleure compréhension sémantique sans nécessiter beaucoup de données.  
- **Exemples :** GloVe, Word2Vec, FastText.  
- **Effet :** améliore la précision et la généralisation du modèle.

### 🧱 3. Régularisation (Dropout, EarlyStopping, ReduceLROnPlateau)
- **Dropout :** désactive aléatoirement des neurones pour éviter le surapprentissage.  
  Exemple : `LSTM(128, dropout=0.3)` coupe 30 % des connexions.  
- **EarlyStopping :** arrête l'entraînement quand la validation n'améliore plus.  
- **ReduceLROnPlateau :** diminue automatiquement le *learning rate* quand la perte stagne.  
- **Effet :** stabilise le modèle et améliore la convergence.

### ✍️ 4. Nettoyage NLP plus riche
| Étape | Action | Exemple |
|-------|---------|----------|
| **Lowercasing** | Mettre tout en minuscules | “Film” → “film” |
| **Suppression de ponctuation** | Enlever “!”, “?”, “.” | “Génial!” → “génial” |
| **Stopwords** | Retirer les mots sans signification | “le”, “de”, “et”… |
| **Lemmatisation** | Ramener à la racine | “jouait” → “jouer” |
| **Tokenisation** | Découper le texte en mots | “ce film est bien” → [“ce”, “film”, “est”, “bien”] |

> 🎙️ À dire : « Le nettoyage du texte est essentiel pour réduire le bruit et aider le LSTM à se concentrer sur le sens. »

### 🔁 5. Cohérence et Reproductibilité
- **Cohérence train/val/test :** appliquer exactement les mêmes transformations aux trois ensembles.  
- **Reproductibilité :** fixer les graines aléatoires et sauvegarder le modèle pour garantir les mêmes résultats à chaque exécution.  
- **Effet :** un entraînement fiable et vérifiable.

> 💬 *À retenir :* Ces techniques rendent le modèle plus robuste, plus cohérent et plus professionnel. Elles constituent les bases d’un pipeline NLP avancé.
