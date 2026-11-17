# TD / TP : Convolutions 1D et 3D dans les CNN

## Plan de la séance

1. Objectifs et rappels
2. Partie TD : exercices papier
3. Partie TP : implémentation PyTorch
4. Éléments de correction

---

## Objectifs pédagogiques

À l'issue de ce TD/TP, l'étudiant(e) doit être capable de :

1. Calculer à la main la sortie d'une convolution 1D et 3D (stride, padding, dilatation).
2. Déterminer les dimensions de sortie, le nombre de paramètres et le coût de calcul (MACs).
3. Calculer le champ réceptif d'un empilement de couches (convolutions dilatées, TCN).
4. Implémenter `conv1d` et `conv3d` from scratch et les valider contre PyTorch.
5. Concevoir, entraîner et évaluer :
   - un CNN 1D pour la classification de séries temporelles ;
   - un CNN 3D pour la classification de volumes (imagerie médicale) ;
6. Comparer Conv3D et la factorisation (2+1)D.

**Prérequis :** Conv2D, rétropropagation, PyTorch (`nn.Module`, `DataLoader`).

---

## Rappel : une seule opération, trois dimensionnalités

Dans les bibliothèques de deep learning, la « convolution » est en réalité une corrélation croisée :

**1D :**

$$y_o[i] = b_o + \sum_{c=1}^{C_{in}} \sum_{k=0}^{K-1} w_{o,c}[k]\, x_c[s\,i + d\,k]$$

**3D :**

$$y_o[t,i,j] = b_o + \sum_{c} \sum_{a=0}^{K_t-1} \sum_{p=0}^{K_h-1} \sum_{q=0}^{K_w-1} w_{o,c}[a,p,q]\, x_c[s_t t + a,\, s_h i + p,\, s_w j + q]$$

($s$ = stride, $d$ = dilatation, $x$ éventuellement complété par du padding).

**Taille de sortie (par axe) :**

$$L_{out} = \left\lfloor \frac{L_{in} + 2p - d(K-1) - 1}{s} \right\rfloor + 1$$

**Nombre de paramètres :**

$$\#\theta = C_{out} \cdot C_{in} \cdot \prod_{axes} K_{axe} + C_{out} \ (\text{biais})$$

---

## Introduction aux couches convolutives

Les couches convolutives (*Convolutional Layers*) constituent des composants fondamentaux des réseaux de neurones convolutifs (*Convolutional Neural Networks* – CNNs).

Elles permettent d'extraire automatiquement des caractéristiques locales à partir des données d'entrée en utilisant des filtres apprenables, également appelés noyaux convolutifs (*Convolutional Kernels*).

Selon la dimensionnalité des données, les opérations convolutives peuvent être réalisées en une, deux ou trois dimensions.

**Remarque importante :**

Dans les bibliothèques de Deep Learning, telles que PyTorch et TensorFlow, l'opération appelée convolution correspond généralement à une corrélation croisée (*Cross-Correlation*).

En effet, les poids du filtre ne sont pas retournés avant leur application sur les données d'entrée. Cette opération est néanmoins désignée par le terme convolution dans les API de ces bibliothèques.

---

## 1. Convolution unidimensionnelle (1D)

La convolution unidimensionnelle est principalement utilisée pour traiter des données organisées sous forme de séquences, telles que les séries temporelles, les signaux et les représentations textuelles.

**Formulation mathématique :**

$$y_o[i] = b_o + \sum_{c=1}^{C_{in}} \sum_{k=0}^{K-1} w_{o,c}[k]\, x_c[si + dk]$$

**Explication des différents termes :**

- $y_o[i]$ : valeur de sortie calculée à la position $i$ dans la carte de caractéristiques (*Feature Map*) associée au filtre $o$.
- $o$ : indice du filtre convolutif. Il identifie le filtre utilisé pour extraire une caractéristique particulière des données.
- $i$ : position à laquelle l'opération de convolution est effectuée dans la représentation de sortie.
- $b_o$ : biais (*Bias*) associé au filtre $o$. Il s'agit d'un paramètre entraînable ajouté au résultat de la convolution.
- $C_{in}$ : nombre de canaux d'entrée (*Input Channels*). Il représente le nombre de canaux présents dans les données d'entrée.
- $c$ : indice du canal d'entrée considéré dans le calcul.
- $K$ : taille du noyau convolutif (*Kernel Size*). Elle représente le nombre de positions couvertes par le filtre.
- $k$ : indice de la position dans le noyau, variant de $0$ à $K-1$.
- $w_{o,c}[k]$ : poids entraînable associé à la position $k$ du noyau, au canal d'entrée $c$ et au filtre de sortie $o$.
- $x_c[si + dk]$ : valeur des données d'entrée sélectionnée en fonction de la position du filtre, du stride et de la dilatation.
- $s$ : pas de déplacement (*Stride*). Il détermine le nombre de positions dont le filtre se déplace entre deux calculs successifs.
- $d$ : facteur de dilatation (*Dilation*). Il contrôle l'espacement entre les éléments successifs du noyau convolutif.

### 1.1. Compréhension du Stride

Le *Stride* représente le nombre de positions dont le filtre convolutif se déplace à chaque étape de calcul.

Par exemple :

- $s = 1$ : le filtre se déplace d'une position à chaque étape.
- $s = 2$ : le filtre se déplace de deux positions à chaque étape.
- $s = 3$ : le filtre se déplace de trois positions à chaque étape.

Un stride élevé réduit généralement la dimension de la représentation de sortie, car le filtre effectue moins de déplacements sur les données d'entrée.

**Interprétation :**

Le stride permet de contrôler la résolution spatiale ou temporelle des caractéristiques extraites par la couche convolutive.

Un stride égal à 1 permet une extraction plus détaillée des caractéristiques locales, tandis qu'un stride supérieur permet de réduire la dimension des représentations intermédiaires.

### 1.2. Compréhension de la Dilatation

La dilatation (*Dilation*) contrôle l'espacement entre les éléments successifs du noyau convolutif.

Elle permet d'augmenter le champ réceptif (*Receptive Field*) sans augmenter le nombre de paramètres entraînables.

Considérons un noyau de taille $K = 3$.

**Cas 1 : Dilatation $d = 1$**

Le filtre utilise trois positions consécutives :

$$[x_1, x_2, x_3]$$

**Cas 2 : Dilatation $d = 2$**

Le filtre utilise les positions suivantes :

$$[x_1, x_3, x_5]$$

**Cas 3 : Dilatation $d = 3$**

Le filtre utilise les positions suivantes :

$$[x_1, x_4, x_7]$$

La taille effective du noyau est donnée par :

$$K_{eff} = d(K-1) + 1$$

Cette expression montre que la dilatation augmente la taille effective du champ couvert par le filtre sans modifier le nombre de poids entraînables.

---

## 2. Convolution tridimensionnelle (3D)

La convolution tridimensionnelle est utilisée pour traiter des données possédant trois dimensions spatiales ou temporelles.

Elle est particulièrement adaptée au traitement des vidéos, des volumes médicaux (IRM, scanners) et des données spatio-temporelles.

**Formulation mathématique :**

$$y_o[t,i,j] = b_o + \sum_{c} \sum_{a=0}^{K_t-1} \sum_{p=0}^{K_h-1} \sum_{q=0}^{K_w-1} w_{o,c}[a,p,q] \times x_c[s_t t + a,\, s_h i + p,\, s_w j + q]$$

**Explication des différents termes :**

- $y_o[t,i,j]$ : valeur de sortie calculée aux positions temporelle $t$, verticale $i$ et horizontale $j$, pour le filtre $o$.
- $o$ : indice du filtre convolutif de sortie.
- $t$ : position temporelle dans la représentation de sortie.
- $i$ : position verticale dans la représentation de sortie.
- $j$ : position horizontale dans la représentation de sortie.
- $b_o$ : biais entraînable associé au filtre de sortie $o$.
- $c$ : indice du canal d'entrée considéré.
- $a$ : indice temporel du noyau convolutif.
- $p$ : indice vertical du noyau convolutif.
- $q$ : indice horizontal du noyau convolutif.
- $K_t$ : taille du noyau sur l'axe temporel.
- $K_h$ : taille du noyau sur l'axe vertical (hauteur).
- $K_w$ : taille du noyau sur l'axe horizontal (largeur).
- $w_{o,c}[a,p,q]$ : poids entraînable associé au filtre $o$, au canal $c$ et aux coordonnées $(a,p,q)$ du noyau.
- $x_c[\cdot]$ : valeur des données d'entrée sélectionnée en fonction des coordonnées temporelles et spatiales.
- $s_t$ : stride temporel, qui détermine le déplacement du filtre sur l'axe temporel.
- $s_h$ : stride vertical, qui détermine le déplacement du filtre sur l'axe de la hauteur.
- $s_w$ : stride horizontal, qui détermine le déplacement du filtre sur l'axe de la largeur.

### 2.1. Exemple d'application à une vidéo

Une vidéo peut être représentée par un tenseur multidimensionnel de la forme :

$$X \in \mathbb{R}^{C_{in} \times T \times H \times W}$$

Où :

- $C_{in}$ : nombre de canaux d'entrée.
- $T$ : nombre d'images (*Frames*) constituant la vidéo.
- $H$ : hauteur de chaque image.
- $W$ : largeur de chaque image.

Considérons un noyau convolutif de dimensions :

$$K_t = 3, \quad K_h = 3, \quad K_w = 3$$

Le filtre analyse simultanément un volume de :

$$3 \times 3 \times 3$$

Cela signifie qu'il examine trois images successives et considère, dans chacune d'elles, une région spatiale de $3 \times 3$ pixels.

Cette opération permet d'extraire des caractéristiques spatio-temporelles, telles que les mouvements, les changements locaux et les interactions entre les images successives.

**Prise en compte de la dilatation en 3D :**

Dans l'équation précédente, la dilatation n'apparaît pas explicitement. Pour généraliser cette formulation, on introduit trois facteurs de dilatation :

$$d_t, \quad d_h, \quad d_w$$

L'équation devient :

$$y_o[t,i,j] = b_o + \sum_{c}\sum_{a}\sum_{p}\sum_{q} w_{o,c}[a,p,q] \times x_c[s_t t + d_t a,\, s_h i + d_h p,\, s_w j + d_w q]$$

---

## 3. Calcul de la taille de sortie

Après l'application d'une opération de convolution, les dimensions des données changent généralement.

La taille de sortie sur un axe donné est calculée à partir de la formule suivante :

$$L_{out} = \left\lfloor \frac{L_{in} + 2p - d(K-1) - 1}{s} \right\rfloor + 1$$

**Explication des différents termes :**

- $L_{out}$ : dimension de sortie obtenue après l'application de la convolution.
- $L_{in}$ : dimension initiale des données avant convolution.
- $p$ : padding, représentant le nombre de valeurs ajoutées de chaque côté de l'axe considéré.
- $d$ : facteur de dilatation, contrôlant l'espacement entre les éléments du noyau.
- $K$ : taille du noyau convolutif sur l'axe considéré.
- $s$ : stride, représentant le pas de déplacement du filtre.
- $\lfloor \cdot \rfloor$ : opérateur de partie entière inférieure. Il permet d'arrondir le résultat vers l'entier inférieur.

### 3.1. Compréhension du Padding

Le *Padding* consiste à ajouter des valeurs supplémentaires autour des données d'entrée, généralement des zéros (*Zero Padding*).

Il permet notamment de contrôler les dimensions de sortie et de préserver les informations situées aux frontières des données.

Deux configurations sont fréquemment utilisées :

**1. Valid Padding**

Aucun remplissage n'est ajouté aux données d'entrée :

$$p = 0$$

Dans ce cas, le filtre ne peut être appliqué que lorsque l'ensemble du noyau reste à l'intérieur des limites des données.

**2. Same Padding**

Le remplissage est choisi de manière à conserver généralement la même dimension spatiale lorsque le stride est égal à 1 :

$$L_{out} = L_{in}$$

Pour un stride égal à 1 et un noyau de taille impaire, le padding symétrique est généralement :

$$p = \frac{K-1}{2}$$

Dans le cas général, notamment lorsque le noyau est pair ou que le stride est supérieur à 1, le padding peut être asymétrique.

### 3.2. Exemple numérique du calcul de sortie

Considérons les paramètres suivants :

$$L_{in} = 32, \quad K = 3, \quad s = 1, \quad d = 1, \quad p = 1$$

En appliquant la formule de calcul de la taille de sortie :

$$L_{out} = \left\lfloor \frac{32 + 2(1) - 1(3-1) - 1}{1} \right\rfloor + 1$$

Après simplification :

$$L_{out} = \left\lfloor \frac{32 + 2 - 2 - 1}{1} \right\rfloor + 1$$

Ainsi :

$$L_{out} = \lfloor 31 \rfloor + 1$$

Finalement :

$$L_{out} = 32$$

La dimension de sortie reste donc identique à celle de l'entrée.

Cet exemple illustre l'utilisation du padding pour conserver la résolution de la représentation après l'opération convolutive.

---

## 4. Nombre de paramètres entraînables

Le nombre total de paramètres d'une couche convolutive correspond au nombre de poids des filtres auxquels on ajoute les biais.

La formule générale est :

$$\#\theta = C_{out} \cdot C_{in} \cdot \prod_{axes} K_{axe} + C_{out}$$

**Explication des différents termes :**

- $\#\theta$ : nombre total de paramètres entraînables de la couche convolutive.
- $C_{out}$ : nombre de canaux de sortie (*Output Channels*). Il correspond au nombre de filtres convolutifs utilisés par la couche.
- $C_{in}$ : nombre de canaux d'entrée (*Input Channels*).
- $K_{axe}$ : taille du noyau sur chacun des axes considérés.
- $\prod_{axes}$ : opérateur produit, qui multiplie les dimensions du noyau convolutif.
- $C_{out}$ (dernier terme) : nombre de biais. Chaque filtre de sortie possède un biais indépendant.

### 4.1. Exemple : Nombre de paramètres en 1D

Considérons une couche convolutive caractérisée par :

$$C_{in} = 4, \quad C_{out} = 16, \quad K = 3$$

**Étape 1 : Calcul du nombre de poids**

Chaque filtre possède $C_{in} \times K$ poids.

Le nombre total de poids est donc :

$$N_{weights} = C_{out} \times C_{in} \times K$$

$$N_{weights} = 16 \times 4 \times 3$$

$$N_{weights} = 192$$

**Étape 2 : Calcul du nombre de biais**

Chaque filtre possède un biais indépendant :

$$N_{bias} = C_{out} = 16$$

**Étape 3 : Calcul du nombre total de paramètres**

$$\#\theta = N_{weights} + N_{bias}$$

$$\#\theta = 192 + 16$$

Finalement :

$$\#\theta = 208$$

La couche convolutive possède donc 208 paramètres entraînables.

### 4.2. Exemple : Nombre de paramètres en 3D

Considérons une couche convolutive possédant les caractéristiques suivantes :

$$C_{in} = 3, \quad C_{out} = 32$$

Avec un noyau de dimensions :

$$K_t = 3, \quad K_h = 3, \quad K_w = 3$$

**Étape 1 : Calcul du nombre de poids**

Le nombre de poids est donné par :

$$N_{weights} = C_{out} \times C_{in} \times K_t \times K_h \times K_w$$

En remplaçant les paramètres :

$$N_{weights} = 32 \times 3 \times 3 \times 3 \times 3$$

$$N_{weights} = 2592$$

**Étape 2 : Calcul du nombre de biais**

Chaque filtre possède un biais :

$$N_{bias} = C_{out} = 32$$

**Étape 3 : Calcul du nombre total de paramètres**

$$\#\theta = 2592 + 32$$

Finalement :

$$\#\theta = 2624$$

La couche convolutive 3D possède donc 2624 paramètres entraînables.

---

## 5. Synthèse générale

Les équations étudiées décrivent quatre aspects fondamentaux du fonctionnement des couches convolutives dans le Deep Learning.

**1. Convolution 1D**

Elle permet d'extraire des caractéristiques à partir de données organisées sous forme de séquences unidimensionnelles.

Ses principaux paramètres sont :

$$K, \quad s, \quad d, \quad w, \quad b$$

**2. Convolution 3D**

Elle permet d'extraire des caractéristiques volumétriques ou spatio-temporelles à partir de données tridimensionnelles.

Ses principaux paramètres sont :

$$K_t, K_h, K_w, s_t, s_h, s_w$$

**3. Calcul de la taille de sortie**

Il permet de déterminer les dimensions des représentations obtenues après l'application d'une convolution.

Les principaux paramètres sont :

$$L_{in}, \quad K, \quad s, \quad d, \quad p$$

**4. Nombre de paramètres entraînables**

Il permet de déterminer le nombre de poids et de biais qui seront optimisés pendant la phase d'apprentissage.

Les principaux paramètres sont :

$$C_{in}, \quad C_{out}, \quad K$$

---

## 6. Remarques importantes

**Distinction entre convolution et corrélation croisée**

Dans les bibliothèques de Deep Learning, l'opération appelée convolution correspond généralement à une corrélation croisée. Les poids du filtre ne sont donc pas retournés avant leur application sur les données d'entrée.

**Influence des paramètres sur la taille de sortie**

Le stride, le padding et la dilatation influencent les dimensions de la représentation de sortie :

- Le stride contrôle le déplacement du filtre.
- Le padding contrôle le remplissage des frontières.
- La dilatation contrôle l'espacement entre les éléments du noyau.

**Influence des paramètres sur le nombre de poids**

Le nombre de paramètres entraînables dépend uniquement :

$$C_{in}, \quad C_{out}, \quad \prod_{axes} K_{axe}$$

Le stride, le padding et la dilatation ne modifient pas le nombre de paramètres entraînables.

Ces distinctions sont essentielles pour comprendre le fonctionnement interne des architectures CNN, analyser leur capacité d'apprentissage et évaluer leur coût computationnel.

---

## Comparatif Conv1D / Conv2D / Conv3D (PyTorch)

|                        | Conv1d                          | Conv2d                    | Conv3d                          |
| ---------------------- | ------------------------------- | ------------------------- | ------------------------------- |
| **Entrée**             | $(N, C, L)$                     | $(N, C, H, W)$            | $(N, C, D, H, W)$               |
| **Poids**              | $(C_o, C_i, K)$                 | $(C_o, C_i, K_h, K_w)$    | $(C_o, C_i, K_d, K_h, K_w)$     |
| **Le noyau glisse sur**| 1 axe                           | 2 axes                    | 3 axes                          |
| **Données typiques**   | audio, ECG, capteurs, texte (embeddings) | images           | vidéos, IRM/CT, nuages voxelisés |
| **Modèles célèbres**   | WaveNet, TCN, InceptionTime     | ResNet, VGG               | C3D, I3D, R(2+1)D, 3D U-Net     |

> **Point clé**
>
> Les canaux ($C$) ne sont jamais un axe de glissement : le noyau couvre toujours la totalité des $C_{in}$ canaux. Une Conv1d sur un ECG 12 dérivations a donc des noyaux de taille $12 \times K$.

---

## Exercice 1 – Convolution 1D à la main

Soit le signal mono-canal et le noyau :

$$x = [\,1,\ 2,\ 0,\ -1,\ 3,\ 1\,], \qquad w = [\,1,\ 0,\ -1\,], \qquad b = 0$$

1. Calculer $y$ pour stride=1, padding=0, dilation=1.
2. Même question avec padding=1 (zéros). Quelle est la taille de sortie ? Commenter (« same » padding).
3. Même question avec stride=2, padding=0.
4. Même question avec dilation=2, padding=0. Quelle est la taille effective du noyau ?
5. Quel type de motif le noyau $w$ détecte-t-il ? (indice : comparer à une dérivée discrète).
6. Vérifier chaque taille de sortie avec la formule générale.

---

## MACs : Multiplications-Accumulations

**Définition :**

Les MACs (*Multiply-Accumulate Operations*) représentent le nombre d'opérations de multiplication-accumulation effectuées par un réseau de neurones pendant son exécution.

Une opération MAC correspond à une multiplication suivie d'une accumulation :

$$a \leftarrow a + (b \times c)$$

Les MACs permettent d'évaluer le coût computationnel d'un modèle de Deep Learning et de comparer l'efficacité de différentes architectures neuronales.

**Objectifs des MACs :**

- Évaluer la complexité computationnelle d'un CNN.
- Comparer l'efficacité de différentes architectures.
- Estimer les ressources nécessaires à l'exécution.
- Étudier les compromis entre précision prédictive et coût computationnel.
- Évaluer l'adaptation des modèles aux dispositifs embarqués.

**Remarque :**

Les paramètres entraînables mesurent la taille du modèle, tandis que les MACs mesurent le travail arithmétique nécessaire à son exécution pour une entrée donnée.

### Calcul des MACs dans une convolution 1D

Considérons une convolution unidimensionnelle :

$$y_o[i] = b_o + \sum_{c=1}^{C_{in}} \sum_{k=0}^{K-1} w_{o,c}[k]\, x_c[si + dk]$$

Pour calculer une seule valeur de sortie, le nombre de multiplications-accumulations est :

$$\text{MACs}_{\text{par sortie}} = C_{in} \times K$$

Puisque la couche possède $C_{out}$ filtres et que chaque filtre produit $L_{out}$ valeurs, le nombre total de MACs est :

$$\text{MACs}_{1D} = L_{out} \times C_{out} \times C_{in} \times K$$

Où :

- $L_{out}$ : longueur de la représentation de sortie.
- $C_{out}$ : nombre de filtres convolutifs.
- $C_{in}$ : nombre de canaux d'entrée.
- $K$ : taille du noyau convolutif.

### Exemple numérique : Convolution 1D

Considérons une couche convolutive caractérisée par :

$$C_{in} = 4, \quad C_{out} = 16, \quad K = 3, \quad L_{out} = 30$$

Le nombre total de MACs est :

$$\text{MACs} = L_{out} \times C_{out} \times C_{in} \times K$$

En remplaçant les paramètres :

$$\text{MACs} = 30 \times 16 \times 4 \times 3$$

Ainsi :

$$\text{MACs} = 5760$$

La couche convolutive nécessite donc 5760 opérations de multiplication-accumulation pour traiter une seule entrée.

### Calcul des MACs dans une convolution 3D

Pour une convolution tridimensionnelle, le nombre de MACs dépend des dimensions temporelles et spatiales de la sortie.

La formule générale est :

$$\text{MACs}_{3D} = T_{out} H_{out} W_{out} \times C_{out} \times C_{in} \times K_t K_h K_w$$

Où :

- $T_{out}$ : nombre de positions temporelles en sortie.
- $H_{out}$ : hauteur de la représentation de sortie.
- $W_{out}$ : largeur de la représentation de sortie.
- $C_{out}$ : nombre de canaux de sortie.
- $C_{in}$ : nombre de canaux d'entrée.
- $K_t$ : taille temporelle du noyau.
- $K_h$ : taille verticale du noyau.
- $K_w$ : taille horizontale du noyau.

### Exemple numérique : Convolution 3D

Considérons une couche convolutive 3D :

$$C_{in} = 3, \quad C_{out} = 32$$

Avec un noyau de dimensions :

$$K_t = 3, \quad K_h = 3, \quad K_w = 3$$

Supposons que les dimensions de sortie soient :

$$T_{out} = 8, \quad H_{out} = 16, \quad W_{out} = 16$$

Le nombre total de MACs est :

$$\text{MACs} = 8 \times 16 \times 16 \times 32 \times 3 \times 3 \times 3 \times 3$$

Finalement :

$$\text{MACs} = 2\,654\,208$$

La convolution 3D nécessite donc environ 2,65 millions de MACs pour traiter une seule entrée.

### Différence entre MACs et FLOPs

Il est important de distinguer les MACs des FLOPs (*Floating-Point Operations*).

Un MAC comprend généralement deux opérations arithmétiques :

- Une multiplication.
- Une addition (accumulation).

Dans la convention couramment utilisée :

$$1 \text{ MAC} \approx 2 \text{ FLOPs}$$

Par conséquent :

$$\text{FLOPs} \approx 2 \times \text{MACs}$$

**Exemple :**

Si une couche convolutive nécessite :

$$\text{MACs} = 1\,000\,000$$

Alors :

$$\text{FLOPs} = 2 \times 1\,000\,000$$

$$\text{FLOPs} = 2\,000\,000$$

**Remarque :**

Certaines publications comptabilisent une opération MAC comme une seule opération, tandis que d'autres comptent séparément la multiplication et l'addition.

Il est donc nécessaire de préciser la convention utilisée pour comparer les résultats expérimentaux.

### Influence des paramètres sur les MACs

Les différents paramètres d'une couche convolutive influencent son coût computationnel.

- $C_{in}$ : une augmentation du nombre de canaux d'entrée augmente le nombre de MACs.
- $C_{out}$ : une augmentation du nombre de filtres augmente le nombre de MACs.
- $K$ : un noyau plus grand augmente le nombre de multiplications-accumulations.
- $L_{out}$ : une sortie plus longue augmente le nombre de MACs.
- Stride : un stride plus élevé réduit généralement les dimensions de sortie et donc les MACs.
- Padding : il influence indirectement les MACs en modifiant les dimensions de sortie.
- Dilation : elle ne modifie pas directement le nombre de MACs lorsque les dimensions de sortie restent identiques.

**Remarque importante :**

Le nombre de paramètres et le nombre de MACs ne sont pas nécessairement proportionnels.

Une même couche peut conserver exactement les mêmes paramètres entraînables, mais nécessiter un nombre différent de MACs selon les dimensions des données d'entrée.

### Calcul des MACs d'un CNN complet

Pour déterminer le coût computationnel global d'un CNN, il faut additionner les MACs de toutes les couches considérées.

$$\text{MACs}_{CNN} = \sum_{l=1}^{L} \text{MACs}_l$$

Où :

- $L$ : nombre de couches considérées.
- $\text{MACs}_l$ : nombre de MACs de la couche $l$.

Pour un CNN composé de convolutions 2D standards :

$$\text{MACs}_{CNN} = \sum_{l=1}^{L} H_l W_l\, C_{out}^{(l)} C_{in}^{(l)} K_h^{(l)} K_w^{(l)}$$

Pour un lot de $B$ observations (*Batch Size*) :

$$\text{MACs}_{Batch} = B \times \text{MACs}_{CNN}$$

**Conclusion :**

Les MACs constituent un indicateur essentiel pour évaluer l'efficacité computationnelle des architectures CNN et analyser le compromis entre performances prédictives et complexité du modèle.

---

## Exercice 2 – Dimensions et paramètres d'un CNN 1D (ECG)

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

1. Donner la forme du tenseur après chaque couche (a) à (e).
2. Calculer le nombre de paramètres de chaque couche et le total.
3. Calculer le nombre de MACs (multiplications-accumulations) de la couche (a) pour un exemple.
4. Pourquoi utiliser `AdaptiveAvgPool1d(1)` plutôt qu'un `Flatten` direct ? Quelle propriété cela donne-t-il au réseau vis-à-vis de la durée du signal ?

---

## Le champ réceptif (Receptive Field – RF)

**Définition :**

Le champ réceptif (*Receptive Field* – RF) désigne la région des données d'entrée qui contribue au calcul de l'activation d'un neurone particulier dans une couche d'un réseau de neurones convolutif (CNN).

Chaque neurone d'une couche convolutive analyse une région locale de l'entrée. Au fur et à mesure que les données traversent plusieurs couches, le champ réceptif augmente progressivement, permettant aux couches profondes de capturer des caractéristiques de plus en plus globales.

**1. Champ réceptif d'une convolution simple**

Pour une convolution unidimensionnelle, la taille effective du champ réceptif d'un filtre est donnée par :

$$RF = d(K-1) + 1$$

Où :

- $RF$ : taille du champ réceptif.
- $K$ : taille du noyau convolutif.
- $d$ : facteur de dilatation (*Dilation*).

Exemple : pour un noyau de taille $K = 3$ :

| $d$ | Positions analysées | $RF$ |
| --- | ------------------- | ---- |
| 1   | $x_1, x_2, x_3$     | 3    |
| 2   | $x_1, x_3, x_5$     | 5    |
| 3   | $x_1, x_4, x_7$     | 7    |

La dilatation permet donc d'augmenter le champ réceptif sans augmenter le nombre de poids entraînables.

### Champ réceptif dans un CNN multicouche

Lorsque plusieurs couches convolutives sont empilées, le champ réceptif augmente progressivement.

Pour calculer le champ réceptif théorique à la couche $l$, on utilise les relations récursives suivantes :

$$RF_l = RF_{l-1} + (K_l - 1)\, d_l\, J_{l-1}$$

$$J_l = J_{l-1}\, s_l$$

Avec les conditions initiales :

$$RF_0 = 1, \quad J_0 = 1$$

**Signification des termes :**

- $RF_l$ : champ réceptif de la couche $l$.
- $RF_{l-1}$ : champ réceptif de la couche précédente.
- $K_l$ : taille du noyau de la couche $l$.
- $d_l$ : facteur de dilatation de la couche $l$.
- $s_l$ : stride de la couche $l$.
- $J_l$ : distance entre deux neurones consécutifs de la couche $l$, exprimée dans les coordonnées de l'entrée initiale.
- $J_{l-1}$ : distance correspondante pour la couche précédente.

Le terme $J_l$ permet de tenir compte de l'effet cumulé des strides des différentes couches.

### Exemple de calcul du champ réceptif

Considérons un réseau CNN composé de trois couches convolutives :

| Couche | $K$ | $s$ | $d$ |
| ------ | --- | --- | --- |
| Conv 1 | 3   | 1   | 1   |
| Conv 2 | 3   | 1   | 1   |
| Conv 3 | 3   | 2   | 1   |

**Première couche :**

$$RF_1 = 1 + (3-1) \times 1 \times 1 = 3$$

$$J_1 = 1 \times 1 = 1$$

**Deuxième couche :**

$$RF_2 = 3 + (3-1) \times 1 \times 1 = 5$$

$$J_2 = 1 \times 1 = 1$$

**Troisième couche :**

$$RF_3 = 5 + (3-1) \times 1 \times 1 = 7$$

$$J_3 = 1 \times 2 = 2$$

Finalement :

$$RF_3 = 7$$

Le neurone de la troisième couche dépend donc théoriquement d'une région de sept positions dans les données d'entrée initiales.

### Importance du champ réceptif

On distingue deux configurations principales :

**1. Champ réceptif local**

Lorsque le champ réceptif est relativement petit, le neurone capture principalement des caractéristiques locales, telles que :

- Les contours et les textures dans les images.
- Les variations locales dans les séries temporelles.
- Les motifs locaux dans les signaux.

**2. Champ réceptif global**

Lorsque le champ réceptif couvre une grande partie ou la totalité des données d'entrée, le neurone peut exploiter des informations provenant de régions éloignées.

Cela permet notamment de capturer :

- Les dépendances à long terme.
- Les relations entre différentes régions d'une image.
- Les interactions entre caractéristiques éloignées.

**Application au Deep Learning :**

Un champ réceptif suffisamment large permet au CNN de prendre en considération un contexte plus global. Son dimensionnement doit néanmoins être adapté à la nature des données et à la tâche d'apprentissage.

### Champ réceptif théorique et effectif

Il est important de distinguer deux notions :

**Champ réceptif théorique (*Theoretical Receptive Field*)**

Il correspond à l'ensemble des positions de l'entrée susceptibles d'influencer l'activation d'un neurone donné.

**Champ réceptif effectif (*Effective Receptive Field*)**

Il représente la contribution réelle des différentes positions de l'entrée à l'activation du neurone.

En pratique, toutes les positions du champ réceptif théorique ne contribuent pas nécessairement avec la même intensité. Les contributions sont souvent concentrées autour du centre du champ réceptif.

**Remarque :**

Un champ réceptif théorique plus large ne garantit pas automatiquement de meilleures performances. Il est nécessaire de choisir une architecture adaptée à la structure des données et aux dépendances que le modèle doit apprendre.

---

## Exercice 3 – Champ réceptif et convolutions dilatées

Le champ réceptif (RF) d'un empilement de couches à stride 1 vérifie :

$$RF_\ell = RF_{\ell-1} + d_\ell (K_\ell - 1), \qquad RF_0 = 1$$

1. Calculer le RF de 3 couches Conv1d avec $K = 3$, $d = 1$.
2. Même question avec les dilatations $d = 1, 2, 4$ (architecture type WaveNet / TCN).
3. Généraliser : RF de $L$ couches de noyau $K$ avec $d_\ell = 2^{\ell-1}$. Combien de couches faut-il pour couvrir 1 seconde d'audio à 16 kHz avec $K = 2$ ?
4. Comparer le nombre de paramètres des deux solutions de la question 1 et 2. Conclure.
5. Causalité : dans un TCN, la sortie à l'instant $t$ ne doit dépendre que de $x_{\le t}$. Comment réaliser ce padding causal avec `nn.Conv1d` ?

---

## Exercice 4 – Convolution 3D à la main

On considère un volume mono-canal $x \in \mathbb{R}^{3 \times 3 \times 3}$ défini par

$$x[d,i,j] = 9d + 3i + j, \qquad d, i, j \in \{0, 1, 2\}$$

(les valeurs vont donc de 0 à 26), et un noyau $w \in \mathbb{R}^{2 \times 2 \times 2}$ rempli de 1, $b = 0$.

1. Quelle est la taille de sortie (stride=1, padding=0) ?
2. Montrer que $y[d,i,j] = 8\,(9d + 3i + j) + 52$.
3. En déduire les 8 valeurs de sortie.
4. À quelle opération classique ce noyau correspond-il (à un facteur près) ?
5. Avec stride=2 et padding=1, quelle est la taille de sortie ?

---

## Exercice 5 – Coût d'un CNN 3D pour la vidéo (type C3D)

On classe des clips vidéo RGB de 16 images en $112 \times 112$ : batch `(8, 3, 16, 112, 112)`.

```python
nn.Conv3d(3, 64, kernel_size=3, padding=1)         # (a)
nn.MaxPool3d(kernel_size=(1, 2, 2))                # (b)
nn.Conv3d(64, 128, kernel_size=3, padding=1)       # (c)
nn.MaxPool3d(kernel_size=2)                        # (d)
```

1. Donner la forme de sortie après (a), (b), (c), (d).
2. Nombre de paramètres de (a) et (c).
3. Nombre de MACs de (a) pour un clip. Comparer avec une Conv2d $3 \to 64$, $3 \times 3$ appliquée image par image sur les 16 images.
4. Pourquoi C3D n'effectue-t-il pas de sous-échantillonnage temporel dans le premier pooling ?
5. Estimer la mémoire (float32) occupée par l'activation de sortie de (a) pour le batch entier.

---

## Intérêt de la factorisation dans les CNN

**Définition :**

La factorisation consiste à remplacer une opération convolutive complexe par plusieurs opérations convolutives plus simples et successives.

L'objectif est de réduire la complexité computationnelle tout en préservant, voire en améliorant, la capacité de représentation du réseau.

Par exemple, une convolution bidimensionnelle utilisant un noyau $3 \times 3$ peut être remplacée par deux convolutions successives :

$$3 \times 3 \;\longrightarrow\; 3 \times 1 \;+\; 1 \times 3$$

La première convolution réalise un filtrage vertical, tandis que la seconde réalise un filtrage horizontal.

Cette décomposition permet de construire un champ réceptif équivalent à celui d'un noyau $3 \times 3$, sous réserve de conserver les dimensions spatiales et d'utiliser un padding adapté.

### 1. Réduction du nombre de paramètres

Considérons une convolution classique avec :

- $C_{in}$ : nombre de canaux d'entrée ;
- $C_{out}$ : nombre de canaux de sortie ;
- $K \times K$ : dimensions du noyau.

Le nombre de paramètres d'une convolution classique est :

$$P_{classique} = K^2 C_{in} C_{out} + C_{out}$$

Le dernier terme correspond aux biais.

Pour un noyau $3 \times 3$ :

$$P_{classique} = 9\, C_{in} C_{out} + C_{out}$$

Dans le cas d'une factorisation en deux convolutions $3 \times 1$ et $1 \times 3$, avec $C_m$ canaux intermédiaires :

$$P_{factoris} = 3\, C_{in} C_m + 3\, C_m C_{out} + C_m + C_{out}$$

Lorsque les nombres de canaux sont identiques :

$$C_{in} = C_m = C_{out} = C$$

On obtient :

$$P_{classique} = 9C^2 + C$$

$$P_{factoris} = 6C^2 + 2C$$

En négligeant les biais lorsque $C$ est suffisamment grand :

$$\frac{P_{factoris}}{P_{classique}} \simeq \frac{6}{9} = \frac{2}{3}$$

La réduction approximative du nombre de paramètres est donc :

$$\text{Réduction} \simeq 33{,}33\,\%$$

### 2. Intérêt du nombre de non-linéarités

L'un des principaux intérêts de la factorisation concerne l'introduction de fonctions d'activation non linéaires entre les convolutions successives.

**Architecture A : convolution classique**

$$Y = \text{ReLU}(W * X + b)$$

Cette architecture contient une seule transformation convolutive suivie d'une fonction d'activation.

**Architecture B : convolution factorisée**

$$Z = \text{ReLU}(W_1 * X + b_1)$$

$$Y = \text{ReLU}(W_2 * Z + b_2)$$

Cette architecture contient deux transformations convolutives et deux fonctions d'activation.

La fonction ReLU est définie par :

$$\text{ReLU}(x) = \max(0, x)$$

### Pourquoi augmenter le nombre de non-linéarités ?

Considérons deux convolutions successives sans fonction d'activation intermédiaire :

$$Z = W_1 * X$$

$$Y = W_2 * Z$$

En utilisant la propriété de composition des convolutions linéaires, on peut écrire :

$$Y = W_2 * (W_1 * X)$$

Cette composition reste une transformation linéaire. Sous certaines conditions, elle peut être représentée par une convolution équivalente.

En revanche, si une fonction d'activation est introduite entre les deux convolutions :

$$Y = W_2 * \text{ReLU}(W_1 * X)$$

La transformation globale devient non linéaire.

Elle ne peut généralement plus être ramenée à une seule convolution linéaire.

**Conséquences :**

- Augmentation de la capacité de représentation du réseau.
- Apprentissage de relations non linéaires plus complexes.
- Construction progressive de caractéristiques intermédiaires.
- Extraction de représentations de plus haut niveau.

### 3. Intérêt de la factorisation pour l'optimisation

La factorisation permet de décomposer une transformation complexe en plusieurs transformations intermédiaires.

$$X \longrightarrow Z_1 \longrightarrow Z_2 \longrightarrow Y$$

Chaque étape apprend une représentation intermédiaire qui contribue à la construction de la représentation finale.

**Principaux avantages :**

1. **Structuration de l'apprentissage :** Le réseau apprend progressivement des transformations intermédiaires au lieu d'apprendre directement une transformation complexe.
2. **Réduction de la complexité :** La diminution du nombre de paramètres peut réduire le coût computationnel et limiter le risque de surapprentissage.
3. **Enrichissement des représentations :** Les fonctions d'activation intermédiaires permettent d'apprendre des transformations non linéaires successives.
4. **Compromis entre complexité et capacité :** La factorisation permet potentiellement d'obtenir une meilleure capacité de représentation avec un nombre réduit de paramètres.

**Remarque :**

La factorisation ne garantit pas systématiquement une optimisation plus rapide ou une meilleure convergence. Son efficacité dépend notamment de l'architecture, de l'initialisation des poids, des fonctions d'activation et de l'algorithme d'optimisation utilisé.

### 4. Exemple numérique

Considérons une convolution avec les paramètres suivants :

$$C_{in} = 64, \quad C_{out} = 64$$

**Cas 1 : convolution classique $3 \times 3$**

$$P_{classique} = 3 \times 3 \times 64 \times 64$$

$$P_{classique} = 36\,864$$

**Cas 2 : convolution factorisée**

On utilise successivement deux convolutions :

$$3 \times 1 \quad \text{et} \quad 1 \times 3$$

avec 64 canaux intermédiaires.

$$P_1 = 3 \times 1 \times 64 \times 64 = 12\,288$$

$$P_2 = 1 \times 3 \times 64 \times 64 = 12\,288$$

Le nombre total de paramètres est :

$$P_{factoris} = P_1 + P_2$$

$$P_{factoris} = 24\,576$$

La réduction obtenue est :

$$\frac{36\,864 - 24\,576}{36\,864} \times 100$$

$$\text{Réduction} = 33{,}33\,\%$$

### 5. Synthèse : convolution classique vs factorisée

| Critère                    | Classique | Factorisée  |
| -------------------------- | --------- | ----------- |
| Nombre de convolutions     | 1         | 2           |
| Nombre de paramètres       | 36 864    | 24 576      |
| Non-linéarités possibles   | 1         | 2           |
| Complexité computationnelle| Plus élevée | Plus faible |
| Représentation             | Directe   | Progressive |

**Conclusion :**

La factorisation présente trois intérêts fondamentaux :

1. **Réduction de la complexité :** diminution du nombre de paramètres et potentiellement du coût computationnel.
2. **Augmentation de la non-linéarité :** introduction de fonctions d'activation intermédiaires permettant d'enrichir la capacité de représentation.
3. **Optimisation :** décomposition de l'apprentissage en plusieurs transformations intermédiaires, pouvant offrir un meilleur compromis entre complexité et performance.

$$\boxed{\text{Factorisation} = \text{Efficacité} + \text{Non-linéarité} + \text{Représentation}}$$

---

## Exercice 6 – Factorisation (2+1)D

Une Conv3d $3 \times 3 \times 3$, $C \to C$ peut être remplacée par :

- une Conv3d spatiale $1 \times 3 \times 3$ : $C \to M$,
- suivie d'une ReLU,
- puis d'une Conv3d temporelle $3 \times 1 \times 1$ : $M \to C$.

1. Exprimer (sans biais) le nombre de paramètres des deux variantes en fonction de $C$ et $M$.
2. Pour $M = C$, calculer le rapport des paramètres.
3. Dans R(2+1)D (Tran et al., 2018), $M$ est choisi pour égaler le nombre de paramètres de la Conv3d. Calculer $M$ en fonction de $C$.
4. Quel est alors l'intérêt de la factorisation ? (indice : nombre de non-linéarités, optimisation).

---

## Exercice 7 – Questions de réflexion

1. Montrer qu'une `Conv1d(C_in, C_out, K)` est équivalente à une `Conv2d` de noyau $(1, K)$ appliquée à une entrée de forme $(N, C_{in}, 1, L)$.
2. Que calcule une Conv1d de noyau $K = 1$ ? Lien avec `nn.Linear` et le bloc feed-forward des Transformers.
3. En NLP, on applique Conv1d sur une phrase de forme $(N, L, E)$ ($E$ = dimension d'embedding). Quelle permutation faut-il faire ? Quel est le rôle des canaux ici ?
4. Une vidéo peut être traitée par (i) Conv2d image par image + agrégation temporelle, (ii) Conv3d, (iii) Conv2d en empilant les $T$ images dans les canaux. Discuter avantages / inconvénients (paramètres, invariance par translation temporelle, coût).
5. Pourquoi une Conv3d est-elle particulièrement adaptée aux scanners CT/IRM par rapport à une Conv2d coupe par coupe ? Quelle difficulté pratique pose-t-elle (mémoire, résolution anisotrope des voxels) ?

---

## TP 1 – Conv1D from scratch (version boucle)

**But :** comprendre exactement ce que fait `F.conv1d`.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def conv1d_naive(x, w, b=None, stride=1, padding=0, dilation=1):
    # x: (N, C_in, L) ; w: (C_out, C_in, K)
    N, C_in, L = x.shape
    C_out, _, K = w.shape
    x = F.pad(x, (padding, padding))
    L_out = (L + 2*padding - dilation*(K - 1) - 1) // stride + 1
    y = x.new_zeros(N, C_out, L_out)
    for i in range(L_out):
        s = i * stride
        patch = x[:, :, s : s + dilation*(K - 1) + 1 : dilation]   # (N, C_in, K)
        y[:, :, i] = torch.einsum('nck,ock->no', patch, w)
    if b is not None:
        y = y + b.view(1, -1, 1)
    return y

# Verification
x = torch.randn(4, 3, 50); w = torch.randn(8, 3, 5); b = torch.randn(8)
for cfg in [dict(stride=1, padding=0, dilation=1),
            dict(stride=2, padding=2, dilation=1),
            dict(stride=1, padding=4, dilation=2)]:
    ok = torch.allclose(conv1d_naive(x, w, b, **cfg), F.conv1d(x, w, b, **cfg), atol=1e-5)
    print(cfg, ok)
```

### TP 1 (suite) – Version vectorisée avec `unfold`

```python
def conv1d_unfold(x, w, b=None, stride=1, padding=0, dilation=1):
    C_out, C_in, K = w.shape
    x = F.pad(x, (padding, padding))
    span = dilation * (K - 1) + 1
    patches = x.unfold(2, span, stride)[..., ::dilation] # (N, C_in, L_out, K)
    y = torch.einsum('nclk,ock->nol', patches, w)
    return y if b is None else y + b.view(1, -1, 1)
```

**Travail demandé :**

1. Vérifier l'égalité avec `F.conv1d` sur les 3 configurations précédentes.
2. Retrouver les résultats de l'Exercice 1 avec votre fonction.
3. Mesurer les temps d'exécution (`time.perf_counter`) des 3 versions pour $L \in \{10^2, 10^3, 10^4, 10^5\}$. Tracer les courbes (échelle log).
4. Bonus : vérifier que `torch.autograd.gradcheck` passe sur `conv1d_unfold` (en float64).

---

## TP 2 – Jeu de données : séries temporelles synthétiques

Trois classes : sinus, carré, dent de scie – fréquence, phase, amplitude et bruit aléatoires.

```python
import math
from torch.utils.data import TensorDataset, DataLoader

def make_dataset(n=3000, L=256, noise=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, L)
    X, Y = [], []
    for _ in range(n):
        c = torch.randint(0, 3, (1,), generator=g).item()
        f = 2 + 8 * torch.rand(1, generator=g)       # 2 a 10 Hz
        ph = 2 * math.pi * torch.rand(1, generator=g)
        A = 0.5 + torch.rand(1, generator=g)
        u = 2 * math.pi * f * t + ph
        if c == 0:   s = torch.sin(u)
        elif c == 1: s = torch.sign(torch.sin(u))
        else:        s = 2 * ((u / (2 * math.pi)) % 1) - 1
        s = A * s + noise * torch.randn(L, generator=g)
        X.append(s); Y.append(c)
    return torch.stack(X).unsqueeze(1), torch.tensor(Y)    # (n, 1, L), (n,)

Xtr, Ytr = make_dataset(3000, seed=0); Xte, Yte = make_dataset(1000, seed=1)
train_dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
test_dl = DataLoader(TensorDataset(Xte, Yte), batch_size=256)
```

**Variante réelle :** ECG5000 ou FordA (archive UCR, `tslearn` / `aeon`).

### TP 2 – Modèle CNN 1D et boucle d'entraînement

```python
class CNN1D(nn.Module):
    def __init__(self, c_in=1, n_classes=3):
        super().__init__()
        def block(ci, co, k):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2),
                                 nn.BatchNorm1d(co), nn.ReLU())
        self.features = nn.Sequential(
            block(c_in, 32, 7), nn.MaxPool1d(2),
            block(32, 64, 5),   nn.MaxPool1d(2),
            block(64, 128, 3), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

def run(model, epochs=15, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
        print(f"epoch {ep:2d} loss {loss.item():.3f} test acc {correct/len(Xte):.3f}")

run(CNN1D())
```

### TP 2 – Travail demandé

1. Entraîner `CNN1D` et reporter l'accuracy de test. Afficher la matrice de confusion.
2. Baseline : comparer à un MLP (entrée aplatie 256 → 128 → 3) à nombre de paramètres comparable. Expliquer l'écart.
3. Robustesse : évaluer les deux modèles sur un jeu de test dont les signaux sont décalés circulairement (`torch.roll`) d'un nombre aléatoire d'échantillons. Conclure sur l'équivariance par translation.
4. Longueur variable : évaluer `CNN1D` (sans ré-entraîner) sur des signaux de longueur 512. Pourquoi est-ce possible ? Pourquoi le MLP ne le peut-il pas ?
5. Bruit : tracer l'accuracy en fonction de `noise` $\in \{0.1, 0.3, 0.6, 1.0\}$.
6. Visualisation : afficher les 32 filtres de la première couche ainsi que les cartes d'activation pour un exemple de chaque classe. Interpréter.
7. Bonus TCN : remplacer les blocs par des convolutions causales dilatées ($d = 1, 2, 4, 8$) avec connexions résiduelles ; comparer champ réceptif, paramètres et performance.

---

## TP 3 – Conv3D from scratch

```python
def conv3d_naive(x, w, b=None, stride=1, padding=0):
    # x: (N, C_in, D, H, W) ; w: (C_out, C_in, Kd, Kh, Kw)
    C_out, C_in, Kd, Kh, Kw = w.shape
    p = padding
    x = F.pad(x, (p, p, p, p, p, p))            # W, H puis D
    N, _, D, H, W = x.shape
    Do, Ho, Wo = (D-Kd)//stride + 1, (H-Kh)//stride + 1, (W-Kw)//stride + 1
    y = x.new_zeros(N, C_out, Do, Ho, Wo)
    for t in range(Do):
        for i in range(Ho):
            for j in range(Wo):
                a, c, e = t*stride, i*stride, j*stride
                patch = x[:, :, a:a+Kd, c:c+Kh, e:e+Kw]          # (N, C_in, Kd, Kh, Kw)
                y[:, :, t, i, j] = torch.einsum('ncdhw,ocdhw->no', patch, w)
    return y if b is None else y + b.view(1, -1, 1, 1, 1)

# Exercice 4
x = torch.arange(27.).view(1, 1, 3, 3, 3)
w = torch.ones(1, 1, 2, 2, 2)
print(conv3d_naive(x, w).flatten())
print(torch.allclose(conv3d_naive(x, w), F.conv3d(x, w)))
```

**À faire :** valider contre `F.conv3d` sur des tenseurs aléatoires (plusieurs canaux, stride 2, padding 1), puis écrire une version vectorisée avec trois `unfold` successifs.

### TP 3 (suite) – Classification de volumes : OrganMNIST3D

**Données :** `pip install medmnist` – volumes CT $1 \times 28 \times 28 \times 28$, 11 organes.

```python
from medmnist import OrganMNIST3D
tr = OrganMNIST3D(split='train', download=True)
te = OrganMNIST3D(split='test', download=True)
to_t = lambda ds: (torch.tensor(ds.imgs, dtype=torch.float32).unsqueeze(1),
                   torch.tensor(ds.labels).squeeze().long())
Xtr, Ytr = to_t(tr); Xte, Yte = to_t(te)          # (N, 1, 28, 28, 28)

class CNN3D(nn.Module):
    def __init__(self, n_classes=11):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1),
                                 nn.BatchNorm3d(co), nn.ReLU(), nn.MaxPool3d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                      nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
```

Sans accès réseau : générer des volumes synthétiques (sphère / cube / cylindre bruités).

### TP 3 – Travail demandé

1. Calculer à la main les formes intermédiaires et le nombre de paramètres de `CNN3D`, puis vérifier avec `torchinfo.summary`.
2. Entraîner 20 époques (Adam, $lr = 10^{-3}$). Reporter accuracy et AUC (macro) de test.
3. Baseline 2D : entraîner une Conv2d sur la coupe centrale uniquement, puis une Conv2d qui traite les 28 coupes comme 28 canaux. Comparer aux résultats 3D.
4. Augmentation 3D : ajouter des flips et rotations de $90^\circ$ aléatoires selon les 3 axes (`torch.flip`, `torch.rot90`). Effet sur la généralisation ?
5. Ressources : mesurer le temps par époque et la mémoire GPU maximale (`torch.cuda.max_memory_allocated`) des modèles 2D et 3D.
6. Interprétabilité (bonus) : implémenter un Grad-CAM 3D sur la dernière couche convolutive et visualiser trois coupes orthogonales de la carte.

---

## TP 4 – Conv3D vs (2+1)D

```python
class Conv2Plus1D(nn.Sequential):
    def __init__(self, ci, co, mid=None):
        mid = mid or (27 * ci * co) // (9 * ci + 3 * co)   # meme nb de parametres
        super().__init__(
            nn.Conv3d(ci, mid, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(),
            nn.Conv3d(mid, co, (3, 1, 1), padding=(1, 0, 0), bias=False))

count = lambda m: sum(p.numel() for p in m.parameters())
print(count(nn.Conv3d(64, 64, 3, bias=False)), count(Conv2Plus1D(64, 64)))
```

1. Vérifier numériquement le résultat de l'Exercice 6.
2. Remplacer les `nn.Conv3d` de `CNN3D` par `Conv2Plus1D` (en conservant BatchNorm/ReLU/Pool). Ré-entraîner et comparer : accuracy, paramètres, temps, courbe de perte d'entraînement.
3. Ouverture vidéo : avec `torchvision.models.video.r2plus1d_18(weights=...)`, extraire des caractéristiques sur quelques clips de UCF101 (ou HMDB51) et entraîner un classifieur linéaire.

---

## Modalités de rendu

**Livrable :** un notebook Jupyter propre et exécutable + un rapport PDF de 4 pages maximum.

Le rapport doit contenir :

- les réponses argumentées aux exercices 1 à 7 (calculs détaillés) ;
- les tableaux de résultats (accuracy, paramètres, temps, mémoire) ;
- les figures demandées (filtres, activations, courbes, Grad-CAM) ;
- une discussion critique : quand préférer Conv1D, Conv2D, Conv3D ou (2+1)D ?

**Reproductibilité :** fixer les graines (`torch.manual_seed`), indiquer les versions des bibliothèques.

| Barème indicatif                                     | Points |
| ---------------------------------------------------- | ------ |
| TD (exercices 1–7)                                   | 6      |
| TP 1 & TP 3 (implémentations from scratch validées)  | 4      |
| TP 2 (CNN 1D + études demandées)                     | 4      |
| TP 3 & TP 4 (CNN 3D, baselines, (2+1)D)              | 4      |
| Qualité du rapport et analyse critique               | 2      |

---

# Éléments de correction

## Correction – Exercice 1

Corrélation croisée : $y[i] = x[i] - x[i+2]$ (stride 1, dilatation 1).

1. **Pas de padding :** $L_{out} = (6 - 2 - 1)/1 + 1 = 4$

   $y = [\,1 - 0,\ 2 - (-1),\ 0 - 3,\ -1 - 1\,] = [\,1,\ 3,\ -3,\ -2\,]$

2. **Padding 1 :** $\tilde{x} = [0, 1, 2, 0, -1, 3, 1, 0]$, $L_{out} = 6$ (« same »)

   $y = [\,-2,\ 1,\ 3,\ -3,\ -2,\ 3\,]$

3. **Stride 2 :** $L_{out} = \lfloor 3/2 \rfloor + 1 = 2$, $y = [\,1,\ -3\,]$

4. **Dilatation 2 :** noyau effectif $d(K-1) + 1 = 5$, $L_{out} = 6 - 5 + 1 = 2$

   $y[i] = x[i] - x[i+4]$, $y = [\,1 - 3,\ 2 - 1\,] = [\,-2,\ 1\,]$

5. $w$ est (au signe près) une dérivée centrée : $y[i] \approx -2\,x'[i+1]$ ; il détecte les fronts montants/descendants.

## Correction – Exercices 2 et 3

**Exercice 2**

- (a) $L = \lfloor (1000 + 6 - 6 - 1)/2 \rfloor + 1 = 500 \Rightarrow (32, 64, 500)$ ; $\#\theta = 64 \cdot 12 \cdot 7 + 64 = 5\,440$
- (b) $(32, 64, 250)$ ; 0 paramètre
- (c) $(32, 128, 250)$ ; $\#\theta = 128 \cdot 64 \cdot 5 + 128 = 41\,088$
- (d) $(32, 128, 1)$ puis Flatten $(32, 128)$ ; (e) $(32, 5)$ ; $\#\theta = 128 \cdot 5 + 5 = 645$
- **Total : 47 173 paramètres.** MACs (a) : $64 \cdot 500 \cdot 12 \cdot 7 = 2\,688\,000$.
- Le pooling global rend le réseau indépendant de la longueur $L$ (et invariant par translation globale).

**Exercice 3**

- $K = 3$, $d = 1$ : $RF = 1 + 2 + 2 + 2 = 7$. Dilatations 1, 2, 4 : $RF = 1 + 2(1 + 2 + 4) = 15$.
- Cas général : $RF = 1 + (K-1)(2^L - 1)$. Avec $K = 2$ : $RF = 2^L \ge 16\,000 \Rightarrow L = 14$ couches.
- Même nombre de paramètres dans les deux cas (même $K$) : la dilatation agrandit le RF gratuitement.
- Padding causal : `F.pad(x, (d*(K-1), 0))` puis `Conv1d(..., padding=0)` (ou padding symétrique puis suppression des $d(K-1)$ derniers pas).

## Correction – Exercices 4 et 5

**Exercice 4**

- Sortie $2 \times 2 \times 2$.
- $y[d,i,j] = \sum_{a,p,q \in \{0,1\}} \big[\,9(d+a) + 3(i+p) + (j+q)\,\big] = 8(9d + 3i + j) + 4 \cdot 9 + 4 \cdot 3 + 4 \cdot 1$

  Soit $8(9d + 3i + j) + 52$ : $y = [\,52,\ 60,\ 76,\ 84,\ 124,\ 132,\ 148,\ 156\,]$ (ordre $d, i, j$)
- Noyau de 1 = $8 \times$ average pooling 3D $2 \times 2 \times 2$ (stride 1).
- Stride 2, padding 1 : $\lfloor (3 + 2 - 2)/2 \rfloor + 1 = 2 \Rightarrow 2 \times 2 \times 2$.

**Exercice 5**

- (a) $(8, 64, 16, 112, 112)$ ; (b) $(8, 64, 16, 56, 56)$ ; (c) $(8, 128, 16, 56, 56)$ ; (d) $(8, 128, 8, 28, 28)$
- $\#\theta$ (a) $= 64 \cdot 3 \cdot 27 + 64 = 5\,248$ ; (c) $= 128 \cdot 64 \cdot 27 + 128 = 221\,312$
- MACs (a) $= 64 \cdot 16 \cdot 112^2 \cdot 81 \approx 1{,}04$ GMAC ; Conv2d image par image : $\times 27/9$ en moins $\approx 0{,}35$ GMAC (3 fois moins).
- On garde la résolution temporelle au début pour ne pas « écraser » trop tôt l'information de mouvement.
- Activation (a) : $8 \cdot 64 \cdot 16 \cdot 112^2 \cdot 4$ octets $\approx 411$ Mo (rien que pour une couche !).

## Correction – Exercices 6 et 7

**Exercice 6**

- Conv3d : $27C^2$. (2+1)D : $9CM + 3MC = 12CM$.
- $M = C$ : rapport $12/27 \approx 0{,}44$ (56 % de paramètres en moins).
- Égalité : $12CM = 27C^2 \Rightarrow M = 2{,}25\,C$ (formule générale : $M = \left\lfloor \dfrac{t\,d^2\,C_{in}\,C_{out}}{d^2\,C_{in} + t\,C_{out}} \right\rfloor$).
- À budget égal : deux fois plus de non-linéarités (fonction plus expressive) et optimisation plus facile (perte d'entraînement plus basse, résultats empiriques de R(2+1)D).

**Exercice 7 (points attendus)**

- Conv1d $\equiv$ Conv2d $(1, K)$ sur une « image » de hauteur 1 (c'est ainsi que certaines bibliothèques l'implémentent).
- $K = 1$ : couche linéaire partagée appliquée à chaque position (= FFN position-wise d'un Transformer).
- NLP : `x.permute(0, 2, 1)` $\to (N, E, L)$ ; les dimensions d'embedding deviennent les canaux.
- Vidéo : (i) ignore le mouvement local, (ii) capture le spatio-temporel mais coûteux, (iii) peu de paramètres mais perd l'équivariance temporelle et fusionne le temps dès la 1re couche.
- Imagerie : contexte 3D anatomique ; difficultés = mémoire, voxels anisotropes (noyaux $1 \times 3 \times 3$, ré-échantillonnage).
