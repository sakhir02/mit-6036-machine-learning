# MIT 6.036 — Introduction to Machine Learning

Implémentations personnelles (NumPy, *from scratch*) des concepts fondamentaux de machine learning, réalisées dans le cadre d'un approfondissement autodidacte du cours **MIT 6.036** (MIT OpenCourseWare), en parallèle de mon Master 1 Mathématiques Appliquées (UPEC).

## Contenu

| Homework | Sujet | Concepts clés |
|---|---|---|
| `hw01_perceptron` | Classification linéaire | Perceptron, perceptron through origin, marge |
| `hw02_features` | Feature engineering | One-hot encoding, features polynomiales |
| `hw03_svm` | Support Vector Machines | Hinge loss, objectif SVM, descente de gradient |
| `hw04_gradient_descent` | Optimisation | Gradient descent, SGD, régression ridge |
| `hw05_neural_networks` | Réseaux de neurones | Forward/backward propagation, modules Linear/ReLU/Tanh/Softmax |

## Approche

Chaque algorithme est d'abord implémenté **sans bibliothèque de ML** (uniquement NumPy), pour comprendre en profondeur :
- la mécanique du calcul du gradient (règle de la chaîne, backpropagation)
- la vectorisation des opérations matricielles
- les choix de conception (fonctions de perte, activations, régularisation)

## Stack technique

- Python 3.10
- NumPy
- Matplotlib (visualisation)

## Lien avec mon projet personnel

Ces fondamentaux ont servi de base à mon projet à venir un système de reconnaissance vocale pour la prise de commande automatisée en restauration, où ces briques (classification, réseaux de neurones) sont étendues avec des modèles pré-entraînés (Whisper, CamemBERT).

## Auteur

Elhadji Mamadou Sakhir Sarr — Master 1 Mathématiques Appliquées, UPEC
