from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import nibabel as nib
from nilearn import plotting, image, datasets
import os
import tkinter as tk
import subprocess
from tkinter import filedialog
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

# Inicializar la aplicación FastAPI
app = FastAPI()

# Directorios para las imágenes procesadas
INFERENCE_FOLDER = "./inference_data"
OUTPUT_2D_FOLDER = "./outputs_2d"
OUTPUT_3D_FOLDER = "./outputs_3d"

# Crear las carpetas si no existen
os.makedirs(INFERENCE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_2D_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_3D_FOLDER, exist_ok=True)

# Función para abrir el explorador de archivos
def open_file_explorer():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii"), ("NIfTI files (gzipped)", "*.nii.gz")])
    return file_path

# Función para mostrar la imagen 3D utilizando OrthoSlicer3D
def show_3d_image(nifti_file_path: str):
    # Cargar el archivo NIfTI
    img = nib.load(nifti_file_path)
    
    # Crear la visualización 3D
    view = plotting.view_img(img, threshold=0.5)
    view.open_in_browser()

@app.get("/visualizar_3d/")
async def visualizar_3d():
    """Abre un explorador de archivos y visualiza la imagen NIfTI en 3D usando OrthoSlicer3D."""
    nifti_file_path = open_file_explorer()

    if nifti_file_path:
        # Ejecutar en un proceso separado
        show_3d_image(nifti_file_path)

        return {"message": f"Visualización 3D abierta para el archivo: {nifti_file_path}"}
    else:
        return JSONResponse(content={"error": "No se seleccionó ningún archivo"}, status_code=400)
        
@app.post("/process_2d/")
def process_2d():
    """Ejecuta el script inferencia_2d.py cuando se presione el botón Procesar 2D"""
    try:
        # Ejecutar el script de inferencia 2D en un proceso separado
        subprocess.run(["python", "inferencia_2d.py"], check=True)
        return {"message": "Procesamiento 2D completado"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error en el procesamiento 2D: {str(e)}"}

@app.post("/process_3d/")
def process_3d():
    """Ejecuta el script inferencia_3d.py cuando se presione el botón Procesar 3D"""
    try:
        subprocess.run(["python", "inferencia_3d.py"], check=True)
        return {"message": "Procesamiento 3D completado"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error en el procesamiento 3D: {str(e)}"}
