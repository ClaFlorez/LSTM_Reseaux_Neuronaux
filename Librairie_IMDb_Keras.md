# 🎬 Librairie IMDb — Jeu de Données pour l’Analyse de Sentiments (Keras / TensorFlow)

## 🧭 Introduction
Le **jeu de données IMDb** (*Internet Movie Database*) est un ensemble de critiques de films largement utilisé pour entraîner et tester des modèles d’**apprentissage automatique** et de **traitement du langage naturel (NLP)**.

Ce dataset est inclus dans la bibliothèque **Keras**, développée par **François Chollet** (Google), et distribuée avec **TensorFlow**.  
Il sert principalement à **l’analyse de sentiments** — c’est-à-dire déterminer si une critique de film est **positive** ou **négative**.

---

## 🧑‍💻 Origine et Création
- **Créé par :** l’équipe Keras / TensorFlow (Google AI)  
- **Basé sur :** les données publiques du site [IMDb.com](https://www.imdb.com)  
- **Auteur principal de Keras :** François Chollet (Google Brain)  
- **But du dataset :** fournir un corpus textuel prêt à l’emploi pour l’expérimentation et l’enseignement des modèles séquentiels (RNN, LSTM, GRU, Transformers…)

---

## 📦 Contenu du Dataset
| Élément | Description |
|----------|-------------|
| **Nombre total d’exemples** | 50 000 critiques de films |
| **Jeu d’entraînement** | 25 000 critiques |
| **Jeu de test** | 25 000 critiques |
| **Tâche** | Classification binaire (sentiment positif ou négatif) |
| **Format** | Données textuelles converties en séquences numériques |
| **Langue** | Anglais |
| **Balance** | 50 % positives / 50 % négatives |

---

## 🧩 Structure des Données
Chaque critique est déjà **prétraitée** et convertie en **séquence d’entiers** représentant les mots du texte.  
Keras fournit aussi un **dictionnaire (`word_index`)** pour relier ces entiers aux mots d’origine.

### Exemple :
```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("Exemple de critique encodée :", X_train[0][:20])
print("Label associé :", y_train[0])
```

📤 Sortie :
```
Exemple de critique encodée : [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, ...]
Label associé : 1
```
Chaque nombre correspond à **un mot** selon un dictionnaire d’index.

---

## 🔤 Le Dictionnaire des Mots
Le mapping entre mots et indices est accessible via :
```python
word_index = imdb.get_word_index()
print(list(word_index.items())[:10])
```

📥 Exemple :
```
[('the', 1), ('and', 2), ('a', 3), ('of', 4), ('to', 5), ('is', 6), ('in', 7), ...]
```

On peut donc **reconstruire la phrase originale** :
```python
index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

def decode_review(encoded_review):
    return " ".join(index_word.get(i, "?") for i in encoded_review)

print(decode_review(X_train[0])[:200])
```

🔍 Les premiers tokens spéciaux :
| Code | Signification |
|-------|----------------|
| `0` | `<PAD>` — remplissage pour uniformiser la longueur |
| `1` | `<START>` — début de la critique |
| `2` | `<UNK>` — mot inconnu |
| `3` | `<UNUSED>` — réservé pour extensions futures |

---

## ⚙️ Paramètres Importants de `load_data()`
```python
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000, skip_top=0, maxlen=None, seed=113)
```

| Paramètre | Description |
|------------|--------------|
| `num_words` | Garde uniquement les *n* mots les plus fréquents |
| `skip_top` | Ignore les mots les plus fréquents (comme “the”, “and”) |
| `maxlen` | Coupe les critiques trop longues |
| `seed` | Graine aléatoire pour la reproductibilité |

---

## 📊 Statistiques
- Longueur moyenne d’une critique : **233 mots**  
- Longueur maximale : **2 494 mots**  
- Taille du vocabulaire brut : **≈ 88 000 mots**  
- Taille typique après `num_words=10000` : **10 000 mots**  
- Répartition équilibrée : 25k positifs / 25k négatifs

---

## 🧠 Utilisation Typique
L’objectif du dataset IMDb est de **tester des architectures séquentielles** :

| Modèle | Utilisation |
|--------|--------------|
| **RNN simple** | Baseline pour apprentissage séquentiel |
| **LSTM / GRU** | Gestion de la mémoire à long terme |
| **CNN 1D** | Détection locale de motifs linguistiques |
| **Transformers** | Contexte bidirectionnel (modèles modernes) |

---

## 🧪 Exemple de Flux Complet
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Chargement
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train = pad_sequences(X_train, maxlen=200)
X_test  = pad_sequences(X_test,  maxlen=200)

# Modèle
model = Sequential([
    Embedding(10000, 128),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
```

---

## 👁️ Visualisation du Dataset (Décodage)
Exemple de critique décodée :
```
<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played ...
```
Label :  
`1` → **Critique positive**

---

## 🔍 Avantages du Dataset IMDb
✅ Facile à charger (intégré à Keras)  
✅ Prétraité (pas besoin de nettoyage complexe)  
✅ Équilibré et annoté  
✅ Idéal pour la **démonstration de modèles NLP**  
✅ Supporte des architectures variées (RNN, LSTM, GRU, CNN, Transformers)

---

## ⚠️ Limites
⚠️ Langue unique : uniquement en anglais.  
⚠️ Jeu de données prétraité (on ne voit pas le texte brut).  
⚠️ Impossible d’ajouter facilement des critiques personnalisées dans la version intégrée.  
⚠️ Pas adapté à des analyses linguistiques profondes (les indices remplacent le vocabulaire original).

---

## 🧭 Conclusion
Le dataset **IMDb** intégré dans **Keras** est un outil pédagogique incontournable pour comprendre les réseaux neuronaux appliqués au texte.  
Il permet d’apprendre les bases du **NLP (Natural Language Processing)** tout en testant des modèles séquentiels puissants comme le **LSTM**.

> 🎓 En résumé : simple, rapide, reproductible et parfait pour l’enseignement.

---

## 📚 Références
- François Chollet, *Keras Documentation – IMDb Dataset*  
- [TensorFlow – IMDb Movie Review Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)  
- [IMDb.com – Internet Movie Database](https://www.imdb.com)

---

**Auteur :** Claud-IA  
📍 Montréal, 2025  
🧠 *Projet de documentation IA – Deep Learning et NLP*
