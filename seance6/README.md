# Séance 6 — Solutions et cours : Convolutions 1D et 3D

Ce document a pour objectif de fournir une explication claire, pédagogique et honnête des solutions des exercices de la séance 6 sur les convolutions 1D et 3D.

Il ne s’agit pas seulement de donner les réponses finales, mais surtout de comprendre pourquoi elles sont correctes.

---

## 1. Objectif de la séance

À la fin de cette séance, l’étudiant ou l’étudiante doit être capable de :

- calculer à la main la sortie d’une convolution 1D ou 3D ;
- comprendre le rôle du stride, du padding et de la dilatation ;
- déterminer la taille de sortie ;
- estimer le nombre de paramètres et le coût de calcul ;
- interpréter l’effet d’un noyau sur un signal ;
- implémenter les opérations avec PyTorch et les valider.

---

## 2. Rappel : définition d’une convolution

Dans les bibliothèques de deep learning, l’opération appelée convolution correspond en réalité à une corrélation croisée :

### Convolution 1D

$$
y_o[i] = b_o + \sum_{c=1}^{C_{in}} \sum_{k=0}^{K-1} w_{o,c}[k] \, x_c[s i + d k]
$$

### Convolution 3D

$$
y_o[t,i,j] = b_o + \sum_c \sum_{a=0}^{K_t-1} \sum_{p=0}^{K_h-1} \sum_{q=0}^{K_w-1} w_{o,c}[a,p,q] \, x_c[s_t t + d_t a,\ s_h i + d_h p,\ s_w j + d_w q]
$$

Les grands concepts à retenir sont :

- K : taille du noyau ;
- s : stride (pas de déplacement) ;
- p : padding (remplissage) ;
- d : dilatation ;
- b : biais.

---

## 3. Formule de la taille de sortie

Pour une dimension donnée :

$$
L_{out} = \left\lfloor \frac{L_{in} + 2p - d(K-1) - 1}{s} \right\rfloor + 1
$$

Cette formule est essentielle. Elle permet de prédire la taille de sortie avant de calculer la convolution.

### Interprétation des paramètres

- L_in : longueur d’entrée ;
- L_out : longueur de sortie ;
- p : padding ajouté à gauche et à droite ;
- d : dilatation (espacement entre les éléments du noyau) ;
- K : taille du noyau ;
- s : stride.

---

## 4. Stride, padding et dilatation

### 4.1. Stride

Le stride est le pas de déplacement du noyau sur l’entrée.

- stride = 1 : déplacement d’une position à la fois ;
- stride = 2 : déplacement de deux positions à la fois ;
- stride = 3 : déplacement de trois positions à la fois.

Plus le stride est grand, plus la sortie est petite.

### 4.2. Padding

Le padding consiste à ajouter des zéros (ou d’autres valeurs) autour de l’entrée.

- valid padding : aucun remplissage, p = 0 ;
- same padding : on choisit p pour garder la dimension de sortie proche de la dimension d’entrée lorsque le stride est 1.

### 4.3. Dilatation

La dilatation agrandit le champ de vision du noyau sans changer son nombre de paramètres.

La taille effective du noyau est :

$$
K_{eff} = d(K-1) + 1
$$

Cela permet d’extraire des informations sur de plus grandes distances dans le signal.

---

## 5. Exercice 1 — Convolution 1D à la main

### Données

$$
x = [1, 2, 0, -1, 3, 1], \quad w = [1, 0, -1], \quad b = 0
$$

Le noyau a donc une taille K = 3.

---

### 5.1. Question 1 : stride = 1, padding = 0, dilation = 1

On applique simplement le noyau sur les fenêtres successives de longueur 3.

Le calcul est :

$$
y[i] = w[0]x[i] + w[1]x[i+1] + w[2]x[i+2]
$$

avec w = [1, 0, -1]. Donc :

$$
y[i] = x[i] - x[i+2]
$$

Calculons les fenêtres :

1. [1, 2, 0] → 1×1 + 0×2 + (-1)×0 = 1
2. [2, 0, -1] → 1×2 + 0×0 + (-1)×(-1) = 2 + 1 = 3
3. [0, -1, 3] → 1×0 + 0×(-1) + (-1)×3 = -3
4. [-1, 3, 1] → 1×(-1) + 0×3 + (-1)×1 = -2

Donc la sortie est :

$$
y = [1, 3, -3, -2]
$$

#### Vérification de la taille

Longueur d’entrée : 6

$$
L_{out} = 6 - 3 + 1 = 4
$$

C’est bien correct.

---

### 5.2. Question 2 : padding = 1 (zéros)

On ajoute un zéro à gauche et à droite :

$$
\tilde{x} = [0, 1, 2, 0, -1, 3, 1, 0]
$$

Le noyau reste de taille 3.

Le nombre de positions de sortie est :

$$
L_{out} = 8 - 3 + 1 = 6
$$

On calcule maintenant :

1. [0, 1, 2] → 0 - 2 = -2
2. [1, 2, 0] → 1 - 0 = 1
3. [2, 0, -1] → 2 - (-1) = 3
4. [0, -1, 3] → 0 - 3 = -3
5. [-1, 3, 1] → -1 - 1 = -2
6. [3, 1, 0] → 3 - 0 = 3

Donc :

$$
y = [-2, 1, 3, -3, -2, 3]
$$

#### Commentaire

Ce cas est un exemple de “same padding” : on garde la même taille qu’avant (6), ce qui est souvent utile pour ne pas diminuer la dimension dans les réseaux de neurones.

---

### 5.3. Question 3 : stride = 2, padding = 0

On déplace le noyau de 2 positions à chaque étape.

Fenêtres :

1. [1, 2, 0] → 1
2. [0, -1, 3] → 0 - 3 = -3

Donc :

$$
y = [1, -3]
$$

#### Vérification de la taille

$$
L_{out} = \left\lfloor \frac{6 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{3}{2} \right\rfloor + 1 = 1 + 1 = 2
$$

Le résultat est bien de taille 2.

---

### 5.4. Question 4 : dilation = 2, padding = 0

La dilatation vaut 2, donc le noyau prend des éléments espacés de 2 positions.

La taille effective du noyau est :

$$
K_{eff} = d(K-1) + 1 = 2(3-1) + 1 = 5
$$

Le noyau effective est donc de taille 5, mais il ne contient que 3 poids, juste espacés.

Les positions utiles sont :

- x[0], x[2], x[4]
- x[1], x[3], x[5]

Cela donne :

1. [1, 0, 3] → 1 - 3 = -2
2. [2, -1, 1] → 2 - 1 = 1

Donc :

$$
y = [-2, 1]
$$

#### Vérification de la taille

$$
L_{out} = 6 - 5 + 1 = 2
$$

C’est correct.

---

### 5.5. Question 5 : quel type de motif détecte le noyau ?

Le noyau est :

$$
w = [1, 0, -1]
$$

Ce noyau compare une valeur du signal avec une autre située deux positions plus loin.

Autrement dit, il calcule une différence locale :

$$
y[i] = x[i] - x[i+2]
$$

C’est un opérateur de détection de variation, presque comme une dérivée discrète.

Il met donc en évidence :

- les fronts montants ;
- les fronts descendants ;
- les changements brusques dans le signal.

En termes simples : il détecte les “bords” ou les changements de tendance dans le signal.

---

### 5.6. Question 6 : vérification avec la formule générale

La formule générale est :

$$
L_{out} = \left\lfloor \frac{L_{in} + 2p - d(K-1) - 1}{s} \right\rfloor + 1
$$

On remplace les valeurs.

#### Cas 1 : stride 1, padding 0, dilatation 1

$$
L_{out} = \left\lfloor \frac{6 + 0 - 1(3-1) - 1}{1} \right\rfloor + 1
$$

$$
= \left\lfloor \frac{6 - 2 - 1}{1} \right\rfloor + 1 = 3 + 1 = 4
$$

#### Cas 2 : padding 1

$$
L_{out} = \left\lfloor \frac{6 + 2 - 2 - 1}{1} \right\rfloor + 1 = 5 + 1 = 6
$$

#### Cas 3 : stride 2

$$
L_{out} = \left\lfloor \frac{6 - 2 - 1}{2} \right\rfloor + 1 = \left\lfloor \frac{3}{2} \right\rfloor + 1 = 1 + 1 = 2
$$

#### Cas 4 : dilatation 2

$$
L_{out} = \left\lfloor \frac{6 - 2*2 - 1}{1} \right\rfloor + 1 = \left\lfloor \frac{6 - 4 - 1}{1} \right\rfloor + 1 = 1 + 1 = 2
$$

Les résultats sont cohérents.

---

## 6. Exercice 2 – Dimensions et paramètres d’un CNN 1D (ECG)

On traite des ECG 12 dérivations échantillonnés à 500 Hz pendant 2 s. Un batch a la forme `(32, 12, 1000)`.

```python
net = nn.Sequential(
    nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),       # (a)
    nn.ReLU(),
    nn.MaxPool1d(2),                                              # (b)
    nn.Conv1d(64, 128, kernel_size=5, padding=2),                 # (c)
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),                                      # (d)
    nn.Flatten(),
    nn.Linear(128, 5))                                            # (e)
```

### 6.1. Forme du tenseur après chaque couche

On utilise la formule standard pour une convolution 1D :

$$
L_{out} = \left\lfloor \frac{L_{in} + 2p - K}{s} \right\rfloor + 1
$$

#### (a) Conv1d(12, 64, kernel_size=7, stride=2, padding=3)

Input : `(32, 12, 1000)`

$$
L_{out} = \left\lfloor \frac{1000 + 2\times 3 - 7}{2} \right\rfloor + 1
= \left\lfloor \frac{999}{2} \right\rfloor + 1 = 499 + 1 = 500
$$

Sortie :

$$
(32, 64, 500)
$$

#### (b) ReLU

Aucune modification de forme :

$$
(32, 64, 500)
$$

#### (c) MaxPool1d(2)

Pool size = 2, stride par défaut = 2.

$$
L_{out} = \left\lfloor \frac{500 - 2}{2} \right\rfloor + 1 = 249 + 1 = 250
$$

Sortie :

$$
(32, 64, 250)
$$

#### (d) Conv1d(64, 128, kernel_size=5, padding=2)

Ici, stride = 1 par défaut.

$$
L_{out} = 250 + 2\times2 - 5 + 1 = 250
$$

Sortie :

$$
(32, 128, 250)
$$

#### (e) ReLU

Aucune modification de forme :

$$
(32, 128, 250)
$$

#### (f) AdaptiveAvgPool1d(1)

La dimension temporelle est réduite à 1 :

$$
(32, 128, 1)
$$

#### (g) Flatten

$$
(32, 128, 1) \rightarrow (32, 128)
$$

#### (h) Linear(128, 5)

$$
(32, 5)
$$

Donc le réseau sort :

$$
(32, 5)
$$

---

### 6.2. Nombre de paramètres de chaque couche

Formule pour une convolution 1D :

$$
\#\text{poids} = C_{out} \times C_{in} \times K
$$

$$
\#\text{biais} = C_{out}
$$

#### (a) Conv1d(12, 64, kernel_size=7)

- poids = $64 \times 12 \times 7 = 5376$
- biais = $64$
- total = $5440$

#### (d) Conv1d(64, 128, kernel_size=5)

- poids = $128 \times 64 \times 5 = 40960$
- biais = $128$
- total = $41088$

#### (h) Linear(128, 5)

- poids = $128 \times 5 = 640$
- biais = $5$
- total = $645$

Total :

$$
5440 + 41088 + 645 = 47173
$$

Donc le nombre total de paramètres est :

$$
47\,173
$$

---

### 6.3. MACs de la couche (a)

Une opération MAC correspond à une multiplication suivie d’une accumulation.

Pour une convolution 1D :

$$
\text{MACs} = L_{out} \times C_{out} \times C_{in} \times K
$$

Ici :

- $L_{out} = 500$
- $C_{out} = 64$
- $C_{in} = 12$
- $K = 7$

Donc :

$$
\text{MACs} = 500 \times 64 \times 12 \times 7 = 2\,688\,000
$$

La couche (a) demande donc :

$$
2\,688\,000 \text{ MACs pour un exemple}
$$

---

### 6.4. Pourquoi utiliser AdaptiveAvgPool1d(1) ?

Sans `AdaptiveAvgPool1d(1)`, on aurait un `Flatten` direct et la dimension de sortie dépendrait de la durée du signal.

Exemple :

- si la durée change, le nombre de neurones après flatten change aussi ;
- donc la couche linéaire finale devrait changer.

Avec `AdaptiveAvgPool1d(1)` :

- on réduit toujours la séquence à une seule position temporelle ;
- la sortie reste de taille fixe : $(128, 1)$ ;
- la couche linéaire finale reçoit toujours 128 caractéristiques.

Cela donne au réseau une propriété très utile :

- il est plus robuste à la durée du signal ;
- il peut traiter des séquences de tailles différentes avec la même architecture ;
- il capture une représentation globale de la séquence.

C’est utile pour les ECG, les séries temporelles, les signaux biologiques, etc.

---

## 7. Résumé des réponses finales

Pour l’exercice 1 :

1. Avec stride 1, padding 0, dilation 1 :
   $$
y = [1, 3, -3, -2]
   $$

2. Avec padding 1 :
   $$
y = [-2, 1, 3, -3, -2, 3]
   $$
   (taille sortie = 6, same padding)

3. Avec stride 2 :
   $$
y = [1, -3]
   $$

4. Avec dilation 2 :
   $$
y = [-2, 1]
   $$
   (taille effective du noyau = 5)

5. Le noyau [1, 0, -1] détecte un changement de signal, c’est-à-dire un contour ou une dérivée discrète.

6. Toutes les tailles de sortie sont vérifiées par la formule générale.

Pour l’exercice 2 :

- formes finales : `(32, 5)`
- paramètres totaux : `47 173`
- MACs de la couche (a) : `2 688 000`
- `AdaptiveAvgPool1d(1)` rend le réseau indépendant de la durée du signal.

---

## 8. Conseils pour bien comprendre les convolutions

Voici les points les plus importants à retenir :

- un noyau n’est pas “magique” ; il extrait un motif particulier ;
- le padding sert à contrôler la taille de sortie ;
- le stride réduit la résolution ;
- la dilatation augmente le champ visuel sans ajouter beaucoup de paramètres ;
- la formule générale est la meilleure façon de vérifier les résultats.

---

## 9. Conclusion

Les convolutions 1D et 3D reposent sur la même idée de base : appliquer un noyau sur des fenêtres locales de données pour extraire des caractéristiques. La différence réside dans la dimension des données et dans le nombre d’axes sur lesquels on se déplace.

Le plus important n’est pas seulement de mémoriser les formules, mais de comprendre la logique behind them :

- comment le noyau se déplace ;
- comment les dimensions changent ;
- comment la valeur du noyau influence le signal ;
- comment le réseau “voit” les informations locales.

C’est ce qui donne du sens aux CNNs dans les applications réelles : séries temporelles, langage, images, vidéos, imagerie médicale, etc.

---

Si besoin, ce document peut être complété avec :

- les solutions des exercices 2, 3, 4, 5, 6 et 7 ;
- des schémas visuels ;
- des exemples PyTorch ;
- une version “cours résumé” beaucoup plus courte à relire avant le contrôle.
