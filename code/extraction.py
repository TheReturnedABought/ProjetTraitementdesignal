import os
import customtkinter as ctk
from PIL import Image

# --- GESTION DE L'IMPORT DU FICHIER TRIE.PY ---
try:
    from trie import detect_layout_from_image
except ImportError:
    print("ATTENTION: Impossible d'importer detect_layout_from_image. Mode simulation activé.")
    def detect_layout_from_image(paths):
        # Simulation d'un retour dictionnaire complexe (comme sur ton screen)
        return [ {'status': 'Unknown keyboard layout', 'detected_keys': 47, 'layout': 'Inconnu'} for _ in paths ]
# ----------------------------------------------


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sélecteur d'Images et Analyse de Clavier")
        self.geometry("1000x700") # Un peu plus large pour bien voir
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # --- CHEMINS ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder_path = os.path.join(current_dir, "..", "data")
        
        # --- LISTES IMPORTANTES ---
        self.checkboxes = []      # (widget_checkbox, chemin_fichier, index_ligne)
        self.result_labels = []   # LISTE SÉPARÉE pour ne supprimer QUE les résultats

        # --- INTERFACE ---
        self.title_label = ctk.CTkLabel(self, text="SÉLECTION DES CLAVIERS", font=("Roboto", 24, "bold"), text_color="#3B8ED0")
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Cadre central défilant
        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Cliquez sur une image pour agrandir")
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Configuration des colonnes du scroll_frame
        self.scroll_frame.grid_columnconfigure(0, weight=0) # Image (fixe)
        self.scroll_frame.grid_columnconfigure(1, weight=0) # Checkbox (fixe)
        self.scroll_frame.grid_columnconfigure(2, weight=1) # Résultat (Prend toute la place restante)

        self.status_label = ctk.CTkLabel(self, text="Sélectionnez des images et cliquez sur VALIDER.", font=("Roboto", 14), text_color="#F8A707")
        self.status_label.grid(row=2, column=0, pady=(0, 5), sticky="ew")

        self.btn_validate = ctk.CTkButton(self, text="VALIDER ET ANALYSER (BATCH)", font=("Roboto", 14, "bold"), height=50, fg_color="#2CC985", hover_color="#229A65", command=self.submit)
        self.btn_validate.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.load_images()


    def load_images(self):
        if not os.path.exists(self.folder_path):
            print(f"ERREUR: Le dossier data est introuvable ici : {self.folder_path}")
            return
        
        extensions = (".png", ".jpg", ".jpeg", ".webp")
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(extensions)]

        for i, filename in enumerate(files):
            file_path = os.path.join(self.folder_path, filename)
            try:
                # 1. Image
                if not os.path.exists(file_path): continue
                pil_img = Image.open(file_path)
                preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(100, 75))
                
                img_label = ctk.CTkLabel(self.scroll_frame, text="", image=preview_image, cursor="hand2")
                img_label.grid(row=i, column=0, padx=10, pady=5)
                img_label.bind("<Button-1>", command=lambda event, p=file_path: self.open_full_image(p))

                # 2. Checkbox
                chk = ctk.CTkCheckBox(self.scroll_frame, text=filename, font=("Roboto", 14))
                chk.grid(row=i, column=1, padx=10, pady=5, sticky="w")
                
                # 3. Stockage (pour ne pas perdre la checkbox)
                self.checkboxes.append((chk, file_path, i))

            except Exception as e:
                print(f"Erreur chargement {filename}: {e}")

    def open_full_image(self, path):
        top = ctk.CTkToplevel(self)
        top.title("Zoom")
        top.geometry("800x600")
        top.attributes("-topmost", True)
        try:
            pil_img = Image.open(path)
            w, h = pil_img.size
            max_w, max_h = 800, 600
            ratio = min(max_w/w, max_h/h)
            new_size = (int(w * ratio * 0.95), int(h * ratio * 0.95))
            full_image = ctk.CTkImage(pil_img, size=new_size)
            ctk.CTkLabel(top, text="", image=full_image).pack(expand=True, fill="both")
        except Exception as e:
            ctk.CTkLabel(top, text=f"Erreur: {e}").pack()


    def submit(self):
        # --- 1. NETTOYAGE CHIRURGICAL ---
        # On supprime seulement les textes de résultats précédents
        # Les checkboxes NE SONT PAS touchées car elles ne sont pas dans cette liste.
        for label in self.result_labels:
            label.destroy()
        self.result_labels = [] 

        # --- 2. RECUPERATION ---
        files_to_analyze = []
        for chk, file_path, row_index in self.checkboxes:
            if chk.get() == 1:
                files_to_analyze.append(file_path)
        
        if not files_to_analyze:
            self.status_label.configure(text="❌ Aucune image sélectionnée.", text_color="#FF0000")
            return
            
        self.status_label.configure(text=f"Analyse en cours de {len(files_to_analyze)} image(s)...", text_color="#F8A707")
        self.update_idletasks()

        # --- 3. ANALYSE ---
        try:
            raw_results = detect_layout_from_image(files_to_analyze)
            
            # Conversion liste -> dictionnaire si besoin
            batch_results = {}
            if isinstance(raw_results, dict):
                batch_results = raw_results
            elif isinstance(raw_results, list):
                if len(raw_results) == len(files_to_analyze):
                    for path, res in zip(files_to_analyze, raw_results):
                        batch_results[path] = res
                else:
                    for i, path in enumerate(files_to_analyze):
                        if i < len(raw_results): batch_results[path] = raw_results[i]

        except Exception as e:
            self.status_label.configure(text=f"Erreur script: {e}", text_color="#FF0000")
            return

        # --- 4. AFFICHAGE JOLI ---
        count = 0
        for chk, file_path, row_index in self.checkboxes:
            if chk.get() == 1:
                raw_data = batch_results.get(file_path, "Erreur")
                
                # NETTOYAGE DU TEXTE (Pour éviter le gros {dict} moche)
                display_text = str(raw_data)
                color = "#2CC985"

                if isinstance(raw_data, dict):
                    # Si c'est un dictionnaire, on essaie de prendre juste le 'status' ou le 'layout'
                    if 'status' in raw_data:
                        display_text = raw_data['status']
                    elif 'layout' in raw_data:
                        display_text = raw_data['layout']
                    
                    # Si c'est "Unknown...", on met en orange/rouge
                    if "Unknown" in display_text or "Echec" in display_text:
                        color = "#F8A707"
                
                # Création du label résultat
                res_lbl = ctk.CTkLabel(self.scroll_frame, 
                                     text=f"➜ {display_text}", 
                                     font=("Roboto", 14, "bold"), 
                                     text_color=color,
                                     wraplength=400, # Important si le texte est long
                                     justify="left")
                
                res_lbl.grid(row=row_index, column=2, padx=10, pady=5, sticky="w")
                
                # IMPORTANT : On ajoute ce label à la liste pour pouvoir le supprimer plus tard
                self.result_labels.append(res_lbl)
                count += 1

        self.status_label.configure(text=f"✅ Terminé ! ({count} résultats)", text_color="#2CC985")


def selectionner_images():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    selectionner_images()