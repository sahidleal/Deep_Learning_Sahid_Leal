"""
Script para descargar el dataset completo de Cats vs Dogs
Descargará aproximadamente 25,000 imágenes
"""

import os
import tensorflow_datasets as tfds
from PIL import Image

# Crear directorio de salida
raw_dir = "/workspaces/Deep_Learning_Sahid_Leal/data/raw/train"
os.makedirs(raw_dir, exist_ok=True)

print("Descargando dataset 'cats_vs_dogs' de TensorFlow Datasets...")
print("Esto puede tomar varios minutos (el dataset tiene ~3 GB)...\n")

# Descargar el dataset. Cats vs Dogs solo proporciona el split train.
train_ds, info = tfds.load(
    'cats_vs_dogs',
    split='train',
    with_info=True,
    as_supervised=True
)

print(f"Dataset info: {info}")
print(f"Total de imágenes en train: {info.splits['train'].num_examples}")
print()

# Descargar y guardar las imágenes
print("Extrayendo y guardando imágenes...\n")

counter = {'cat': 0, 'dog': 0}

for image, label in train_ds:
    # label: 0 = cat, 1 = dog
    label_name = 'cat' if label == 0 else 'dog'
    
    # Convertir tensor a imagen PIL
    image_np = image.numpy()
    image_pil = Image.fromarray(image_np)
    
    # Generar nombre de archivo
    filename = f"{label_name}.{counter[label_name]}.jpg"
    filepath = os.path.join(raw_dir, filename)
    
    # Guardar imagen
    image_pil.save(filepath, quality=95)
    
    counter[label_name] += 1
    
    # Mostrar progreso cada 500 imágenes
    total = counter['cat'] + counter['dog']
    if total % 500 == 0:
        print(f"Procesadas {total} imágenes... (Gatos: {counter['cat']}, Perros: {counter['dog']})")

total_images = counter['cat'] + counter['dog']
print(f"\n✅ ¡Descarga completada!")
print(f"Total de imágenes guardadas: {total_images}")
print(f"  - Gatos: {counter['cat']}")
print(f"  - Perros: {counter['dog']}")
print(f"Ubicación: {raw_dir}")
