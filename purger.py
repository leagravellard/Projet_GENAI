import shutil
import os

DB_PATH = "chroma_db"

# Supprime complètement le dossier de la base
shutil.rmtree(DB_PATH, ignore_errors=True)

# Vérification simple
if not os.path.exists(DB_PATH):
    print(f"{DB_PATH} a été purgé avec succès.")
else:
    print(f"Impossible de purger {DB_PATH}.")