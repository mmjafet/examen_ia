import pandas as pd
import numpy as np
import cv2
import random
import base64
from flask import Flask, render_template
import os

# Inicializar Flask y Talisman para la política de seguridad
app = Flask(__name__)

# Cargar los datos
brain_df = pd.read_csv('Brain_MRI/data_mask.csv')

@app.route('/')
def index():
    # Seleccionamos un índice aleatorio
    i = random.randint(0, len(brain_df) - 1)

    # Cargar imágenes
    mri_image_path = os.path.join('Brain_MRI', brain_df.iloc[i]['image_path'])
    mri_image = cv2.imread(mri_image_path)
    mri_image = cv2.cvtColor(mri_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB

    mask_image_path = os.path.join('Brain_MRI', brain_df.iloc[i]['mask_path'])
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Superponer la máscara sobre la imagen MRI
    mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
    combined_image = cv2.addWeighted(mri_image, 0.7, mask_colored, 0.3, 0)

    # Convertir imágenes a base64
    _, buffer_mri = cv2.imencode('.png', mri_image)
    mri_base64 = base64.b64encode(buffer_mri).decode()

    _, buffer_mask = cv2.imencode('.png', mask_image)
    mask_base64 = base64.b64encode(buffer_mask).decode()

    _, buffer_combined = cv2.imencode('.png', combined_image)
    combined_base64 = base64.b64encode(buffer_combined).decode()

    return render_template('index.html', mri_image=mri_base64, mask_image=mask_base64, combined_image=combined_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)