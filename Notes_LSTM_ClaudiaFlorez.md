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

# 🧮 Les Formules du LSTM (Long Short-Term Memory)

Le LSTM est une version améliorée du réseau récurrent (RNN) qui permet de **mémoriser des informations sur de longues séquences**.  
Il utilise trois *portes principales* — oubli, entrée et sortie — pour gérer le flux d’informations.

À chaque étape temporelle \( t \), le modèle reçoit :
- \( x^{(t)} \) : l’entrée actuelle (par exemple, un mot)
- \( h^{(t-1)} \) : la sortie précédente (mémoire courte)
- \( c^{(t-1)} \) : l’état de la cellule précédente (mémoire longue)

---

## 🟧 1. Porte d’Oubli (*Forget Gate*)

Décide quelles informations de la mémoire précédente \( c^{(t-1)} \) doivent être **supprimées ou conservées**.

\[
f^{(t)} = \sigma(W_f \cdot [h^{(t-1)}, x^{(t)}] + b_f)
\]

- \( f^{(t)} \) prend des valeurs entre 0 et 1 :  
  - 0 → oubli total  
  - 1 → conservation complète

---

## 🟩 2. Porte d’Entrée (*Input Gate*)

Contrôle **quelle nouvelle information** doit être ajoutée à la mémoire.

\[
i^{(t)} = \sigma(W_i \cdot [h^{(t-1)}, x^{(t)}] + b_i)
\]
\[
\tilde{c}^{(t)} = \tanh(W_c \cdot [h^{(t-1)}, x^{(t)}] + b_c)
\]

- \( i^{(t)} \) : décide combien de la nouvelle information sera intégrée  
- \( \tilde{c}^{(t)} \) : vecteur de **nouvelles valeurs candidates** à ajouter à la mémoire

---

## 🧱 3. Mise à Jour de la Mémoire (*Cell State Update*)

Combine l’ancienne mémoire \( c^{(t-1)} \) et la nouvelle pour former l’état actualisé \( c^{(t)} \) :

\[
c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}
\]

où \( \odot \) représente la **multiplication élément par élément** (*Hadamard product*).

---

## 🟦 4. Porte de Sortie (*Output Gate*)

Décide **quelle partie de la mémoire** sera visible dans la sortie finale \( h^{(t)} \).

\[
o^{(t)} = \sigma(W_o \cdot [h^{(t-1)}, x^{(t)}] + b_o)
\]
\[
h^{(t)} = o^{(t)} \odot \tanh(c^{(t)})
\]

- \( o^{(t)} \) : filtre la sortie  
- \( h^{(t)} \) : sortie réelle du LSTM (et entrée du pas suiva


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
