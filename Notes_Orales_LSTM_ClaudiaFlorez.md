# Notes orales — Les Réseaux LSTM  
### Présentation de Claudia Florez  
_Intelligence Artificielle, 2025_  

---

## 🧠 Introduction
Les réseaux de neurones classiques traitent les données de manière indépendante, sans mémoire du passé.  
Les **Réseaux de Neurones Récurrents (RNN)** ont introduit la capacité de conserver une trace temporelle, mais ils souffrent d’un problème majeur : **l’oubli du contexte à long terme**.

> Les LSTM (Long Short-Term Memory), proposés par *Hochreiter et Schmidhuber (1997)*, ont été conçus pour surmonter cette limite.

---

## ⚠️ Le problème des RNN classiques
Les RNN standards peuvent se souvenir de quelques pas temporels, mais pas de longues séquences.  
Lorsqu’une dépendance se trouve loin dans le temps, les gradients s’atténuent — c’est le **vanishing gradient problem**.

> Exemple : pour prédire la phrase “Je parle français”, le modèle doit se souvenir du mot “France” mentionné bien plus tôt.

Cette incapacité à gérer les dépendances longues limite leur performance dans le traitement du langage, la voix ou les séries temporelles.

---

## 💡 L’idée du LSTM
Le **LSTM** introduit un **mécanisme de mémoire contrôlée** permettant au réseau d’apprendre *quoi retenir et quoi oublier*.  
Son cœur est la **cellule mémoire** (*cell state*), un flux d’information principal, modulé par trois portes :

1. **Porte d’oubli** (*forget gate*) : supprime les informations inutiles.  
2. **Porte d’entrée** (*input gate*) : ajoute les nouvelles informations pertinentes.  
3. **Porte de sortie** (*output gate*) : décide ce qui est transmis à la sortie.

> Ces portes utilisent des fonctions sigmoïdes (valeurs entre 0 et 1) pour contrôler le flux d’information.  

---

## ⚙️ Fonctionnement étape par étape

### 🔹 1. Porte d’oubli
La porte d’oubli choisit quelles parties de la mémoire passée \(C_{t-1}\) doivent être effacées :  
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

### 🔹 2. Porte d’entrée
Elle détermine quelles nouvelles informations \(\tilde{C}_t\) seront ajoutées :  
$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$  
$$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

### 🔹 3. Mise à jour de la mémoire
La cellule mémoire est mise à jour selon :  
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$

### 🔹 4. Porte de sortie et état caché
Enfin, la sortie est calculée :  
$$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$  
$$ h_t = o_t * \tanh(C_t) $$

> Ces formules assurent que l’information utile est conservée pendant de longues séquences, tout en évitant l’explosion ou la disparition des gradients.

---

## 🚀 Applications et variantes

### 🌍 Domaines d’application
- Traduction automatique (Google Translate, DeepL)  
- Reconnaissance vocale (Siri, Alexa)  
- Prédiction de séries temporelles (finance, météo, santé)  
- Génération de texte (anciens modèles GPT, analyse de sentiments)

### 🔧 Variantes du LSTM
- **Peephole LSTM** : les portes consultent l’état de la cellule.  
- **GRU (Gated Recurrent Unit)** : version simplifiée combinant les portes d’entrée et d’oubli.  
- **Coupled Forget/Input Gates** : réduction du nombre de paramètres.  

> Malgré ces variantes, les performances restent comparables (Greff et al., 2015).

---

## 📘 Conclusion — Vers les Transformers
Les **LSTM** ont marqué une étape cruciale dans le deep learning.  
Ils ont permis aux réseaux de **stabiliser l’apprentissage séquentiel** et de **mémoriser sur le long terme**.

Aujourd’hui, les modèles **Transformers** ont remplacé les LSTM dans de nombreux domaines grâce au **mécanisme d’attention**, qui apprend les dépendances globales entre tous les éléments d’une séquence.

> Les LSTM restent néanmoins essentiels pour comprendre l’évolution des architectures séquentielles modernes.  

---

**Résumé final :**
- Les RNN = mémoire courte.  
- Les LSTM = mémoire longue contrôlée.  
- Les Transformers = mémoire globale avec attention.  

✨ _Les LSTM sont la passerelle entre le passé des RNN et l’avenir des Transformers._
