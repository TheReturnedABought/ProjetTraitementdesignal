#extraction
import os
import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sélecteur d'Images")
        self.geometry("800x600")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # --- MODIFICATION IMPORTANTE ICI ---
        # On récupère le chemin du dossier où se trouve ce fichier (code/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # On remonte d'un cran (..) pour aller chercher 'data'
        self.folder_path = os.path.join(current_dir, "..", "data")
        # -----------------------------------

        self.checkboxes = []
        self.selected_files = []

        self.title_label = ctk.CTkLabel(self, text="SÉLECTION DES CLAVIERS", font=("Roboto", 24, "bold"), text_color="#3B8ED0")
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Cliquez sur une image pour agrandir")
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(1, weight=1)

        self.load_images()

        self.btn_validate = ctk.CTkButton(self, text="VALIDER", font=("Roboto", 14, "bold"), height=50, fg_color="#2CC985", hover_color="#229A65", command=self.submit)
        self.btn_validate.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

    def load_images(self):
        if not os.path.exists(self.folder_path):
            print(f"ERREUR: Le dossier data est introuvable ici : {self.folder_path}")
            return
        
        extensions = (".png", ".jpg", ".jpeg", ".webp")
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(extensions)]

        for i, filename in enumerate(files):
            file_path = os.path.join(self.folder_path, filename)
            try:
                pil_img = Image.open(file_path)
                preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(100, 75))
                
                img_label = ctk.CTkLabel(self.scroll_frame, text="", image=preview_image, cursor="hand2")
                img_label.grid(row=i, column=0, padx=10, pady=5)
                img_label.bind("<Button-1>", command=lambda event, p=file_path: self.open_full_image(p))
            except: pass

            chk = ctk.CTkCheckBox(self.scroll_frame, text=filename, font=("Roboto", 14))
            chk.grid(row=i, column=1, padx=10, pady=5, sticky="w")
            self.checkboxes.append((chk, file_path))

    def open_full_image(self, path):
        top = ctk.CTkToplevel(self)
        top.title("Zoom")
        top.geometry("800x600")
        top.attributes("-topmost", True)
        try:
            pil_img = Image.open(path)
            w, h = pil_img.size
            ratio = min(800/w, 600/h)
            new_size = (int(w * ratio), int(h * ratio))
            full_image = ctk.CTkImage(pil_img, size=new_size)
            ctk.CTkLabel(top, text="", image=full_image).pack(expand=True, fill="both")
        except: pass

    def submit(self):
        self.selected_files = [path for chk, path in self.checkboxes if chk.get() == 1]
        self.quit()
        self.destroy()

def selectionner_images():
    app = App()
    app.mainloop()
    return app.selected_files

if __name__ == "__main__":
    print(selectionner_images())