

![image](https://github.com/user-attachments/assets/e5105a23-c385-4117-80eb-d43e7c163a91)



disclamer : "OCR output" donne tous les caractères identifiées mais par soucis de place, il n'affiche que les 1O premiers (qui sont souvent les touches Fx ou les numéros) donc même si le résultat est AZERTY et que la sortie OCR est pas du tout cohérente, c'est normal.


### Keyboard Layout Detector (AZERTY / QWERTY)

Ce projet détecte automatiquement le layout de clavier (AZERTY ou QWERTY) à partir d’images de clavier en utilisant OpenCV, EasyOCR et un système de scoring basé sur les rangées de touches.

### 1. Fonctionnalités

- Upscale automatique de l’image pour améliorer l’OCR.
- Plusieurs pré-traitements d’image (contraste, flou, sharpen, inversion).
- OCR multi‑méthodes avec EasyOCR.
- Clustering des touches par rangée (haut / milieu / bas) avec GaussianMixture (scikit‑learn).
- Scoring avancé des layouts (AZERTY, QWERTY) à partir :
    - des caractères détectés,
    - de leur rangée estimée,
    - des caractères indicateurs / conflits spécifiques à chaque layout.

**Visualisation complète** :

- image originale,
- images pré‑traitées avec bounding boxes,
- résumé du layout détecté + confiance.

### 2. Installation
**2.1. Cloner le dépôt**

´´´
bash
git clone <URL_REPO>
cd <VOTRE_REPO>
´´´

**2.2. Créer un environnement virtuel (recommandé)**
´´´
bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# ou
.\.venv\Scripts\activate       # Windows
´´´

**2.3. Installer les dépendances**
´´´
bash
pip install -r requirements.txt
´´´

Contenu du requirements.txt :
- text
- opencv-python
- numpy
- matplotlib
- easyocr
- scikit-learn
  
Remarque : EasyOCR installera automatiquement une version CPU de PyTorch. Pour un GPU, installe manuellement la bonne roue PyTorch avant EasyOCR.

**3. Structure principale du code**

/project
│── detect_layout.py       # pipeline principal
│── utils.py               # fonctions de prétraitement & dessin
│── extraction.py          # interface CustomTkinter
│── data/                  # images de claviers
│── requirements.txt
│── README.md


### 4. Utilisation
Lancer 
´python main.py´

Fonctionnalités de l’interface :
- aperçu des images (scrollable + zoom)
- sélection multiple
- analyse batch
- affichage clair des résultats

### 5. Formats d’images supportés
Formats standards OpenCV : jpg, jpeg, png…

Les images doivent contenir le clavier en entier, idéalement vu de dessus, bien éclairé. Le code effectue un upscale interne (upscale_image) donc les photos de smartphone fonctionnent bien.

### 6. Limitations / pistes d’amélioration

- Résultats sensibles à la qualité d’éclairage et au contraste des légendes de touches.
- EasyOCR peut encore produire des confusions sur certains caractères (ex : 0/O, 1/I, 5/S), partiellement corrigées par LIKELY_KEYMAP.
- Pour des layouts exotiques ou des claviers très compacts, il peut être utile :
    - d’ajuster les allowlists OCR,
    - de tuner les poids dans score_layout_with_row_clustering,
    - d’ajouter des layouts dans LAYOUTS.
      
 - EasyOCR lit du bruit comm si c'était une touche

   Pistes d'amélioration :
   - Avoir les photos dans les mêmes conditions (luminosité, angles, résolution, etc.)
   - Utiliser un OCR plus poussé
   - Affiné les pré-traitements le plus possible 

### 7. Licence
MIT License
