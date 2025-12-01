import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# =============================
#  PRÉTRAITEMENT: 3 méthodes optimisées
# =============================

def method1_adaptive_threshold(img):
    img_upscaled = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    inverted = cv2.bitwise_not(gray)

    binary = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=3
    )

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel_erode, iterations=1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_dilate, iterations=1)

    return binary


def method2_lab_channel(img):
    img_upscaled = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    l_channel = cv2.fastNlMeansDenoising(l_channel, h=8)

    _, text_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    text_black_on_white = cv2.bitwise_not(text_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_black_on_white = cv2.morphologyEx(text_black_on_white, cv2.MORPH_CLOSE, kernel)

    return text_black_on_white


def method3_simple_inversion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    inverted = cv2.bitwise_not(gray)
    return inverted


# =============================
#  OCR avec EasyOCR
# =============================

def ocr_easyocr(reader, img, method_name="", min_confidence=0.25):
    results = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        batch_size=1,
        min_size=5,
        text_threshold=0.4,
        low_text=0.2,
        link_threshold=0.2,
        canvas_size=4500,
        mag_ratio=3.0,
        slope_ths=0.3,
        ycenter_ths=0.7,
        height_ths=0.7,
        width_ths=0.7,
        add_margin=0.15,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    )

    detected = [(text, conf) for (bbox, text, conf) in results if conf > min_confidence]

    print(f"\n{'=' * 60}")
    print(f"{method_name}")
    print(f"{'=' * 60}")
    print(f"📊 {len(results)} détections totales")
    print(f"✅ {len(detected)} valides (conf > {min_confidence:.0%})\n")

    for text, conf in detected:
        print(f"  ✓ '{text}' ({conf:.1%})")

    return results, detected


# =============================
#  COMBINAISON DES RÉSULTATS
# =============================

def combine_results(detected_list):
    all_chars = []
    for detected in detected_list:
        for text, conf in detected:
            if len(text) <= 2:
                all_chars.append(text.upper())

    char_counts = Counter(all_chars)
    validated = [c for c, count in char_counts.items() if count >= 2]
    all_unique = list(set(all_chars))

    return validated, all_unique, char_counts


# =============================
#  FILTRAGE AZERTY
# =============================

def filter_azerty_keys(chars):
    valid = set('AZERTYUIOPQSDFGHJKLMWXCVBN0123456789')
    return [c for c in chars if c in valid]


# =============================
#  DÉTECTION AZERTY vs QWERTY
# =============================

def detect_keyboard_layout(detected_chars, verbose=True):
    """
    Détecte si le clavier est AZERTY, QWERTY ou QWERTZ.
    Utilise les caractères de la première ligne avec marge d'erreur.

    Args:
        detected_chars: Liste des caractères détectés
        verbose: Afficher les détails du calcul

    Returns:
        tuple: (layout_name, confidence_score, details_dict)
    """
    # Définition des layouts (première ligne de lettres)
    layouts = {
        'AZERTY': {
            'row1': set('AZERTYUIOP'),
            'indicators': ['A', 'Z'],  # Caractères clés AZERTY
            'conflicts': ['Q', 'W']  # Absents de la 1ère ligne AZERTY
        },
        'QWERTY': {
            'row1': set('QWERTYUIOP'),
            'indicators': ['Q', 'W'],
            'conflicts': ['A', 'Z']  # A et Z pas en 1ère ligne QWERTY
        },
        'QWERTZ': {
            'row1': set('QWERTZUIOP'),
            'indicators': ['Z'],  # Z remplace Y
            'conflicts': ['Y']  # Y absent de la 1ère ligne QWERTZ
        }
    }

    detected_set = set([c.upper() for c in detected_chars if len(c) == 1])

    scores = {}
    details = {}

    for layout_name, layout_info in layouts.items():
        # Score basé sur les caractères de la première ligne
        row1_matches = detected_set & layout_info['row1']
        row1_score = len(row1_matches)

        # Bonus pour les indicateurs clés
        indicator_bonus = sum(2 for char in layout_info['indicators'] if char in detected_set)

        # Pénalité pour les conflits (caractères qui ne devraient pas être là)
        conflict_penalty = sum(3 for char in layout_info['conflicts'] if char in detected_set)

        # Score final
        final_score = row1_score + indicator_bonus - conflict_penalty

        scores[layout_name] = final_score
        details[layout_name] = {
            'row1_matches': row1_matches,
            'row1_score': row1_score,
            'indicator_bonus': indicator_bonus,
            'conflict_penalty': conflict_penalty,
            'final_score': final_score
        }

    # Déterminer le gagnant
    best_layout = max(scores, key=scores.get)
    best_score = scores[best_layout]

    # Calculer la confiance (en %)
    total_detected = len(detected_set)
    max_possible_score = 10  # 10 touches sur la première ligne
    confidence = min(100, (best_score / max_possible_score) * 100) if max_possible_score > 0 else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print("🔍 DÉTECTION DU LAYOUT CLAVIER")
        print(f"{'=' * 60}")
        print(f"Caractères détectés: {sorted(detected_set)}")
        print(f"Total: {total_detected} caractères\n")

        for layout_name, detail in details.items():
            print(f"{layout_name}:")
            print(f"  ├─ Touches 1ère ligne: {detail['row1_matches']} → score: {detail['row1_score']}")
            print(f"  ├─ Bonus indicateurs: +{detail['indicator_bonus']}")
            print(f"  ├─ Pénalité conflits: -{detail['conflict_penalty']}")
            print(f"  └─ Score final: {detail['final_score']}")

        print(f"\n{'=' * 60}")
        print(f"🎯 RÉSULTAT: {best_layout}")
        print(f"📊 Confiance: {confidence:.1f}%")

        # Warnings
        if confidence < 40:
            print("⚠️  Confiance faible - pas assez de caractères détectés")
        elif confidence < 70:
            print("⚠️  Confiance moyenne - vérifier la détection")
        else:
            print("✅ Haute confiance")

        print(f"{'=' * 60}\n")

    return best_layout, confidence, details


# =============================
# VISUALISATION
# =============================

def visualize_results(img_original, processed_images, ocr_results, method_names):
    n = len(processed_images)

    fig = plt.figure(figsize=(20, 12))

    plt.subplot(3, n + 1, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("ORIGINAL", fontsize=11)
    plt.axis('off')

    for i, (proc, name, (results, detected)) in enumerate(zip(
            processed_images, method_names, ocr_results
    )):
        plt.subplot(3, n + 1, i + 2)
        plt.imshow(proc, cmap='gray')
        color = "green" if len(detected) > 0 else "red"
        plt.title(f"{name}\n{len(detected)} det.", fontsize=10, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("ocr_comparison_complete.png", dpi=200, bbox_inches="tight")
    plt.show()


# =============================
# MAIN PIPELINE
# =============================

if __name__ == "__main__":

    print("=" * 60)
    print("🚀 PIPELINE OCR + DÉTECTION LAYOUT CLAVIER")
    print("=" * 60)

    reader = easyocr.Reader(['en', 'fr'], gpu=False)

    img_path = r"../data/02ktahxbanzc1.jpg"
    img_original = cv2.imread(img_path)

    if img_original is None:
        raise FileNotFoundError(f"❌ Image introuvable: {img_path}")

    print(f"✔ Image chargée ({img_original.shape[1]}×{img_original.shape[0]})")

    print("\n🛠 Prétraitements...")
    img_m1 = method1_adaptive_threshold(img_original)
    img_m2 = method2_lab_channel(img_original)
    img_m3 = method3_simple_inversion(img_original)

    print("\n🔍 OCR EasyOCR...")
    r1, d1 = ocr_easyocr(reader, img_m1, "Méthode 1")
    r2, d2 = ocr_easyocr(reader, img_m2, "Méthode 2")
    r3, d3 = ocr_easyocr(reader, img_m3, "Méthode 3")

    print("\n📊 Fusion des résultats...")
    validated, all_unique, counts = combine_results([d1, d2, d3])

    print("\n🧹 Filtrage AZERTY...")
    validated = filter_azerty_keys(validated)
    all_unique = filter_azerty_keys(all_unique)

    print("\nCaractères validés (≥2 méthodes):", validated)
    print("Tous caractères uniques:", all_unique)

    # NOUVELLE FONCTIONNALITÉ: Détection du layout
    layout, confidence, details = detect_keyboard_layout(all_unique, verbose=True)

    print("\n📈 Visualisation...")
    visualize_results(
        img_original,
        [img_m1, img_m2, img_m3],
        [(r1, d1), (r2, d2), (r3, d3)],
        ["Méthode 1", "Méthode 2", "Méthode 3"]
    )

    print(f"\n{'=' * 60}")
    print("📋 RÉSUMÉ FINAL")
    print(f"{'=' * 60}")
    print(f"Layout détecté: {layout} ({confidence:.1f}% confiance)")
    print(f"Caractères: {sorted(all_unique)}")
    print(f"{'=' * 60}")
