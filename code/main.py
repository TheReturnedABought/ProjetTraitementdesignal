#main
import skimage
from matplotlib import pyplot as plt
from extraction import selectionner_images
from trie import detect_layout_from_image


def main():
    print("--- Programme Principal ---")
    

    images = selectionner_images()

    if not images:
        print("Aucune image sélectionnée.")
    else:
        print(f"\n{len(images)} images prêtes pour l'analyse :")
        for img_path in images:
            print(f" -> {img_path}")

    print(detect_layout_from_image(images, True))

if __name__ == "__main__":
    main()
