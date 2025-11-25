from extraction import selectionner_images

def main():
    print("--- Programme Principal ---")
    

    images = selectionner_images()

    if not images:
        print("Aucune image sélectionnée.")
    else:
        print(f"\n{len(images)} images prêtes pour l'analyse :")
        for img_path in images:
            print(f" -> {img_path}")
            
        # ICI : Tu appelles ta future fonction de détection
        # detecter_azerty_qwerty(images)
            detect_layout_from_image(img_path)
if __name__ == "__main__":
    main()
