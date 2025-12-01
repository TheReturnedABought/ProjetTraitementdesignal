### Keyboard Layout Detector (AZERTY / QWERTY)

Ce projet détecte automatiquement le layout de clavier (AZERTY ou QWERTY) à partir d’images de clavier en utilisant OpenCV, EasyOCR et un système de scoring basé sur les rangées de touches.

### 1. Fonctionnalités
Upscale automatique de l’image pour améliorer l’OCR.

Plusieurs pré-traitements d’image (contraste, flou, sharpen, inversion).

OCR multi‑méthodes avec EasyOCR.

Nettoyage et fusion des caractères détectés (validation par apparition sur plusieurs méthodes).

Clustering des touches par rangée (haut / milieu / bas) avec GaussianMixture (scikit‑learn).

Scoring avancé des layouts (AZERTY, QWERTY, QWERTZ) à partir :

des caractères détectés,

de leur rangée estimée,

des caractères indicateurs / conflits spécifiques à chaque layout.

**Visualisation complète** :

image originale,

images pré‑traitées avec bounding boxes,

résumé du layout détecté + confiance.

### 2. Installation
**2.1. Cloner le dépôt**
bash
git clone <TON_URL_REPO>
cd <TON_REPO>

**2.2. Créer un environnement virtuel (recommandé)**
bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# ou
.\.venv\Scripts\activate       # Windows

**2.3. Installer les dépendances**
bash
pip install -r requirements.txt
Contenu du requirements.txt :

text
opencv-python
numpy
matplotlib
easyocr
scikit-learn
Remarque : EasyOCR installera automatiquement une version CPU de PyTorch. Pour un GPU, installe manuellement la bonne roue PyTorch avant EasyOCR.

**3. Structure principale du code**
Le cœur du projet se trouve dans ton script (par exemple detect_layout.py) et contient :

Pré‑traitement / utils

upscale_image(...)

increase_contrast(...)

sharpen_image(...)

apply_gaussian_blur(...)

convert_to_gray(...)

draw_boxes_on_image(...)

Méthodes de pré‑traitement

method1_contrast_and_sharpen

method2_blur_and_sharpen

method3_simple_inversion

method4_upscaled_contrast_blur_and_sharpen

OCR & fusion

ocr_keyboard_layout(reader, processed_images)

mapping des erreurs typiques via LIKELY_KEYMAP et _map_likely(...)

Clustering des rangées

get_row_labels_for_validated_chars(...) (GaussianMixture, 3 rangées)

assign_rows_to_chars(...)

Scoring des layouts

définitions LAYOUTS (QWERTY, AZERTY, QWERTZ)

score_layout_with_row_clustering(detected_letters, cluster_labels)

Visualisation

visualize_results(...) (grille 2×N avec images + textes OCR)

Pipeline principal

detect_layout_from_image(img_path, use_row_clustering=True, debug=True)

### 4. Utilisation
**4.1. Depuis un script Python**
python
from detect_layout import detect_layout_from_image

results = detect_layout_from_image(
    img_path="chemin/vers/ton_clavier.jpg",
    use_row_clustering=True,
    debug=True
)

for path, (layout, confidence) in results.items():
    print(f"{path} -> {layout} ({confidence:.1f}%)")
    
**4.2. Depuis la ligne de commande (exemple)**
Si ton fichier s’appelle detect_layout.py et contient un bloc if __name__ == "__main__": qui appelle detect_layout_from_image, tu peux lancer :

bash
python detect_layout.py chemin/vers/ton_clavier.jpg
Le script :

charge l’image,

applique les 4 pré‑traitements,

lance EasyOCR sur chaque version,

agrège / valide les caractères,

estime les rangées,

score chaque layout,

affiche la figure de visualisation avec les bounding boxes et le résumé.

### 5. Formats d’images supportés
Formats standards OpenCV : jpg, jpeg, png…

Les images doivent contenir le clavier en entier, idéalement vu de dessus, bien éclairé.

Le code effectue un upscale interne (upscale_image) donc les photos de smartphone fonctionnent bien.

### 6. Limitations / pistes d’amélioration
Résultats sensibles à la qualité d’éclairage et au contraste des légendes de touches.

EasyOCR peut encore produire des confusions sur certains caractères (ex : 0/O, 1/I, 5/S), partiellement corrigées par LIKELY_KEYMAP.

Pour des layouts exotiques ou des claviers très compacts, il peut être utile :

d’ajuster les allowlists OCR,

de tuner les poids dans score_layout_with_row_clustering,

d’ajouter des layouts dans LAYOUTS.

### 7. Licence
Ajoute ici la licence de ton choix (MIT, Apache‑2.0, etc.), par exemple :

text
MIT License
Copyright (c) 2025 …
