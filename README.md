# G-CNA Neural Network Library

Ce dépôt contient la documentation et les sources de la bibliothèque G-CNA, une collection d'opérateurs, couches et optimisateurs conçus pour expérimenter des réseaux de neurones faits maison. L'objectif est de proposer des blocs de construction simples, compréhensibles et faciles à étendre pour tester des idées d'apprentissage profond sans dépendre d'une stack trop lourde.

## Structure du dépôt

- `docs/` : pages de référence organisées par sujet (modules, tenseurs, couches, etc.). C'est ici que vous pouvez lire les explications détaillées.
- `Module/`, `Optimizer/`, `Tensor/`, `Configuration/`, `Helper/` : implémentations de base réparties par responsabilité. Chaque dossier contient le code qui accompagne la documentation correspondante.
- `README.md` : ce fichier d'embarquement rapide.

## Documentation

- [Guide général](docs/DOCUMENTATION.md) : aperçu global des concepts, conventions et flux de travail de la bibliothèque.
- [Modules](docs/modules.md) : classe de base `Module`, structure `Sequential` et étapes de calcul vers la propagation avant.
- [Tenseurs et paramètres](docs/tensor.md) : `Tensor`, `Parameter` et rappel rapide de l'autodiff.
- [Couches](docs/layers.md) : couches `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten` et leurs usages.
- [Activations](docs/activations.md) : fonctions d'activation (`ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`).
- [Pertes](docs/losses.md) : implémentations de `MSE`, `L1`, `CrossEntropy`, `BCE`.
- [Optimiseurs](docs/optimizers.md) : `SGD`, `Adam` et planificateur `StepLr`.
- [Opérations de rétropropagation](docs/backward_ops.md) : topo des opérations générées pendant la rétroprop.
- [Helpers](docs/helper.md) : initialiseurs, techniques de pruning et quantification.
- [Analyse](docs/analyzer.md) : outils d'inspection et de débogage de flux.

## Premiers pas

1. Explorez les fichiers de `docs/` pour comprendre les primitives disponibles.
2. Étudiez les dossiers `Module/`, `Tensor/`, etc. pour voir les implémentations réelles.
3. Ajoutez vos propres modules ou extensions dans les dossiers existants et créez ou mettez à jour une page dans `docs/` si nécessaire.
