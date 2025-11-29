# main.py

import skimage
# from matplotlib import pyplot as plt # Plus nécessaire
from extraction import selectionner_images
# from trie import detect_layout_from_image # Plus nécessaire ici

def main():
    print("--- Programme Principal ---")
    
    # La fonction sélectionne et lance l'interface
    # L'analyse se fait dans l'interface lorsque l'utilisateur clique sur "VALIDER"
    selectionner_images() 
    
    # Le programme se termine lorsque l'utilisateur ferme la fenêtre.
    print("Fermeture du programme.")

if __name__ == "__main__":
    main()