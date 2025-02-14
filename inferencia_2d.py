import nibabel as nib
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
import shutil
from natsort import natsorted
from skimage.transform import resize
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes, apply_orientation, inv_ornt_aff
from glob import glob

def porcentaje(numero, porcentaje):
    num = numero * (porcentaje/100)
    return int(num)

def save_slices(datos, paciente, media, desviacion):
    slice_axial = []
    slice_coronal = []
    slice_sagital = []
    
    slice_nombres_axial = []
    slice_nombres_coronal = []
    slice_nombres_sagital = []
    
    num_index = datos.shape[2]
    inicio = porcentaje(num_index, 20)
    fin = num_index - porcentaje(num_index, 20)

    for index in range(inicio, fin):
        slice_a = datos[:, :, index]
        if np.any(slice_a != 0):
            slice_a = (slice_a - media) / desviacion
            slice_a = np.interp(slice_a, (slice_a.min(), slice_a.max()), (0, 255)).astype(np.uint8)
        slice_axial.append(slice_a)
        slice_nombres_axial.append(f'{paciente}_slice_{index-inicio}.png')

    num_index = datos.shape[0]
    inicio = porcentaje(num_index, 20)
    fin = num_index - porcentaje(num_index, 20)

    for index in range(inicio, fin):
        slice_s = datos[index, :, :]
        if np.any(slice_s != 0):
            slice_s = (slice_s - media) / desviacion
            slice_s = np.interp(slice_s, (slice_s.min(), slice_s.max()), (0, 255)).astype(np.uint8)
        slice_sagital.append(slice_s)
        slice_nombres_sagital.append(f'{paciente}_slice_{index-inicio}.png')

    num_index = datos.shape[1]
    inicio = porcentaje(num_index, 20)
    fin = num_index - porcentaje(num_index, 20)

    for index in range(inicio, fin):
        slice_c = datos[:, index, :]
        if np.any(slice_c != 0):
            slice_c = (slice_c - media) / desviacion
            slice_c = np.interp(slice_c, (slice_c.min(), slice_c.max()), (0, 255)).astype(np.uint8)
        slice_coronal.append(slice_c)
        slice_nombres_coronal.append(f'{paciente}_slice_{index-inicio}.png')
    
    return slice_axial, slice_coronal, slice_sagital, slice_nombres_axial, slice_nombres_coronal, slice_nombres_sagital

carpeta_pacientes = 'inference_data/'

slice_axial = []
slice_coronal = []
slice_sagital = []

slice_nombres_axial = []
slice_nombres_coronal = []
slice_nombres_sagital = []

global_mean = 102.47110714474171 
global_std = 251.90855992838803

for archivo in natsorted(os.listdir(carpeta_pacientes)):
    ruta = os.path.join(carpeta_pacientes, archivo)
    
    imagen = nib.load(ruta)
    datos = imagen.get_fdata()
    
    fichero = ruta.split('/')[-1]
    paciente = fichero.split('_')[0]
    
    lista1, lista2, lista3, lista4, lista5, lista6 =save_slices(datos, paciente, global_mean, global_std)
    
    slice_axial.extend(lista1)
    slice_coronal.extend(lista2)
    slice_sagital.extend(lista3)
    slice_nombres_axial.extend(lista4)
    slice_nombres_coronal.extend(lista5)
    slice_nombres_sagital.extend(lista6)
    
class Custom_Dataset(Dataset):
    def __init__(self, imagenes, transform=None):
        self.imagenes = imagenes
        self.transform = transform
        
    def __len__(self):
        return len(self.imagenes)
        
    def __getitem__(self, idx):
        img = (self.imagenes[idx])

        target_height = 224
        target_width = 224

        img = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_AREA)
        
        img = img/255.0

        img = np.expand_dims(img, axis=-1)
        
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1).float()

        return img_tensor
    
def crear_lista_pacientes_test(carpeta):
    pruebas = []
    for foto in os.listdir(carpeta):
        paciente = foto.split('_')[0]
        if paciente not in pruebas:
            pruebas.append(paciente)
    return natsorted(pruebas)

def crear_diccionario_pruebas_test(pruebas):
    return {clave: [] for clave in pruebas}

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("UNet_mejor_Axial.pth"))
test_dataset = Custom_Dataset(slice_axial)
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

carpeta = 'inference_data/'
pruebas = crear_lista_pacientes_test(carpeta)
diccionario_axial = crear_diccionario_pruebas_test(pruebas)

model = model.to(device)
model.eval()

for idx, images in enumerate(dataloader_test):
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()
    
    original = images.squeeze().cpu().numpy()
    prediccion = predictions.squeeze().cpu().numpy()

    filename = slice_nombres_axial[idx]
    paciente = os.path.basename(filename).split('_')[0]

    if paciente in diccionario_axial:
        diccionario_axial[paciente].append(prediccion)

lista_nifti_axial = []
for paciente, imagenes in diccionario_axial.items():
  
    prueba = [f for f in os.listdir(carpeta) if paciente in f][0]
    ruta = os.path.join(carpeta, prueba)
    imagen_nifti = nib.load(ruta)
    datos = imagen_nifti.get_fdata()
    
    num_index = datos.shape[2]
    num_añadir = porcentaje(num_index, 20)

    original_height, original_width = datos.shape[:2]

    imagenes_redimensionadas = np.zeros((num_index, original_height, original_width))
    for idx, img in enumerate(imagenes):
        img_resized = resize(img, (original_height, original_width), mode='constant', preserve_range=True)
        img_resized = (img_resized > 0.5).astype(np.float32)
        imagenes_redimensionadas[idx + num_añadir] = img_resized

    pred_masks_np = np.stack(imagenes_redimensionadas, axis=-1)
    
    nifti_img = nib.Nifti1Image(pred_masks_np, np.eye(4))
    
    lista_nifti_axial.append(nifti_img)
    
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("UNet_mejor_Coronal.pth"))
test_dataset = Custom_Dataset(slice_coronal)
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

carpeta = 'inference_data/'
pruebas = crear_lista_pacientes_test(carpeta)
diccionario_coronal = crear_diccionario_pruebas_test(pruebas)

model = model.to(device)
model.eval()

for idx, images in enumerate(dataloader_test):
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()
    
    original = images.squeeze().cpu().numpy()
    prediccion = predictions.squeeze().cpu().numpy()

    filename = slice_nombres_coronal[idx]
    paciente = os.path.basename(filename).split('_')[0]

    if paciente in diccionario_coronal:
        diccionario_coronal[paciente].append(prediccion)

lista_nifti_coronal = []
for paciente, imagenes in diccionario_coronal.items():
    prueba = [f for f in os.listdir(carpeta) if paciente in f][0]
    ruta = os.path.join(carpeta, prueba)
    imagen_nifti = nib.load(ruta)
    datos = imagen_nifti.get_fdata()

    num_index = datos.shape[1]
    num_añadir = porcentaje(num_index, 20)

    original_depth, original_width = datos.shape[0], datos.shape[2]

    imagenes_redimensionadas = np.zeros((num_index, original_depth, original_width))
    for idx, img in enumerate(imagenes):
        img_resized = resize(img, (original_depth, original_width), mode='constant', preserve_range=True)
        img_resized = (img_resized > 0.5).astype(np.float32)
        imagenes_redimensionadas[idx + num_añadir] = img_resized

    pred_masks_np = np.stack(imagenes_redimensionadas, axis=1)

    nifti_img = nib.Nifti1Image(pred_masks_np, np.eye(4))

    lista_nifti_coronal.append(nifti_img)
    
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("UNet_mejor_Sagital.pth"))
test_dataset = Custom_Dataset(slice_sagital)
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

carpeta = 'inference_data/'
pruebas = crear_lista_pacientes_test(carpeta)
diccionario_sagital = crear_diccionario_pruebas_test(pruebas)

model = model.to(device)
model.eval()

for idx, images in enumerate(dataloader_test):
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()
    
    original = images.squeeze().cpu().numpy()
    prediccion = predictions.squeeze().cpu().numpy()

    filename = slice_nombres_sagital[idx]
    paciente = os.path.basename(filename).split('_')[0]

    if paciente in diccionario_sagital:
        diccionario_sagital[paciente].append(prediccion)

lista_nifti_sagital = []
for paciente, imagenes in diccionario_sagital.items():

    prueba = [f for f in os.listdir(carpeta) if paciente in f][0]
    ruta = os.path.join(carpeta, prueba)
    imagen_nifti = nib.load(ruta)
    datos = imagen_nifti.get_fdata()

    num_index = datos.shape[0]
    num_añadir = porcentaje(num_index, 20)

    original_height, original_depth = datos.shape[1], datos.shape[2]

    imagenes_redimensionadas = np.zeros((num_index, original_height, original_depth))
    for idx, img in enumerate(imagenes):
        img_resized = resize(img, (original_height, original_depth), mode='constant', preserve_range=True)
        img_resized = (img_resized > 0.5).astype(np.float32)
        imagenes_redimensionadas[idx + num_añadir] = img_resized

    pred_masks_np = np.stack(imagenes_redimensionadas, axis=0)

    nifti_img = nib.Nifti1Image(pred_masks_np, np.eye(4))
    
    lista_nifti_sagital.append(nifti_img)

def reorient_to_las(nifti_img):
    """ Reorienta una imagen NIfTI a LAS y actualiza la matriz affine correctamente. """
    current_ornt = axcodes2ornt(aff2axcodes(nifti_img.affine))
    target_ornt = axcodes2ornt(('L', 'A', 'S'))  # Orientación deseada: LAS
    transform = ornt_transform(current_ornt, target_ornt)

    # Transformar los datos de la imagen
    reoriented_data = apply_orientation(nifti_img.get_fdata(), transform)

    # Calcular la nueva affine ajustada a la orientación
    new_affine = nifti_img.affine @ inv_ornt_aff(transform, nifti_img.shape)

    # Crear la nueva imagen NIfTI con datos corregidos y affine actualizada
    return nib.Nifti1Image(reoriented_data, affine=new_affine, header=nifti_img.header)

def obtener_niftis_ordenados(carpeta):
    # Buscar todos los archivos .nii y .nii.gz en la carpeta
    nifti_files = natsorted(glob(os.path.join(carpeta, "*.nii*")))  
    
    # Cargar los archivos NIfTI
    lista_niftis = [nib.load(f) for f in nifti_files]
    
    return lista_niftis
    
def umbralizacion(lista_axial, lista_coronal, lista_sagital, slice_nombres_axial, output_dir, lista_nifti_originales):
    os.makedirs(output_dir, exist_ok=True)
    
    pacientes = natsorted(set(nombre.split('_')[0] for nombre in slice_nombres_axial))
    
    print(len(lista_axial), len(lista_coronal), len(lista_sagital), len(lista_nifti_originales))
    
    for i in range(len(lista_axial)):
        paciente = pacientes[i]
        paciente_dir = os.path.join(output_dir, paciente)
        os.makedirs(paciente_dir, exist_ok=True)
        
        nifti_axial = lista_axial[i]
        nifti_coronal = lista_coronal[i]
        nifti_sagital = lista_sagital[i]
        nifti_original = lista_nifti_originales[i]
        
        voxel_data_list = [nifti_axial.get_fdata(), nifti_coronal.get_fdata(), nifti_sagital.get_fdata()]
        
        suma_voxeles = np.sum(voxel_data_list, axis=0)
        
        umbralizado = (suma_voxeles > 1).astype(np.float32)
        
        umbralizado_nifti = nib.Nifti1Image(umbralizado, affine=nifti_original.affine, header=nifti_original.header)
        
        reoriented_pred = reorient_to_las(umbralizado_nifti)
        
        output_filepath = os.path.join(paciente_dir, f"{paciente}_PRED.nii.gz")
        nib.save(reoriented_pred, output_filepath)

output_dir = "outputs_2d/"
carpeta = "inference_data/"
lista_nifti_originales = obtener_niftis_ordenados(carpeta)
umbralizacion(lista_nifti_axial, lista_nifti_coronal, lista_nifti_sagital, slice_nombres_axial, output_dir, lista_nifti_originales)

