import os
import shutil
import nibabel as nib
from natsort import natsorted
import subprocess

def gestion_nifti(source_folder, output_folder):

    global_mean = 102.47110714474171
    global_std = 251.90855992838803
    
    counter = 100
    
    for file in natsorted(os.listdir(source_folder)):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            img_path = os.path.join(source_folder, file)
            img = nib.load(img_path)
            data = img.get_fdata()
            
            standardized_data = (data - global_mean) / global_std
            
            standardized_img = nib.Nifti1Image(standardized_data, img.affine, img.header)
            
            standardized_img_path = os.path.join(output_folder, f'BRATS_{counter:03d}_0000.nii.gz')
            nib.save(standardized_img, standardized_img_path)
            
            counter += 1
    
source_folder = "inference_data/"
output_folder = "nnUNet/nnUNet_raw/Dataset022_Esclerosis/imagesTs/"

os.environ["nnUNet_raw"] = "/home/guillermoclaro/Documentos/TFG/Web/nnUNet/nnUNet_raw/"
os.environ["nnUNet_preprocessed"] = "/home/guillermoclaro/Documentos/TFG/Web/nnUNet/nnUNet_preprocessed/"
os.environ["nnUNet_results"] = "/home/guillermoclaro/Documentos/TFG/Web/nnUNet/nnUNet_results/"

gestion_nifti(source_folder, output_folder)

def ejecutar_prediccion_nnUNet():
    try:
        subprocess.run([
            "nnUNetv2_predict",
            "-i", "/home/guillermoclaro/Documentos/TFG/Web/nnUNet/nnUNet_raw/Dataset022_Esclerosis/imagesTs/",
            "-o", "/home/guillermoclaro/Documentos/TFG/Web/nnUNet/nnUNet_raw/Dataset022_Esclerosis/predictions/",
            "-d", "22",
            "-c", "3d_fullres",
            "-f", "1"
        ], check=True)
        print("Predicción completada con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"Error en la predicción: {str(e)}")
        
ejecutar_prediccion_nnUNet()

def obtener_nombres_pacientes(input_folder):
    pacientes = []
    
    for file in os.listdir(input_folder):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            paciente = file.split('_')[0]
            pacientes.append(paciente) 
    return natsorted(set(pacientes))

def obtener_archivos_predicciones(predictions_folder):
    archivos = [
        file for file in os.listdir(predictions_folder) 
        if file.endswith('.nii.gz')
    ]
    return natsorted(archivos)

def copiar_y_renombrar_predicciones(pacientes, predicciones, predictions_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(pacientes)):
        paciente = pacientes[i]
        prediccion_original = os.path.join(predictions_folder, predicciones[i])
        
        paciente_folder = os.path.join(output_folder, paciente)
        os.makedirs(paciente_folder, exist_ok=True)
        
        prediccion_nueva = os.path.join(paciente_folder, f"{paciente}_PRED.nii.gz")

        shutil.copy(prediccion_original, prediccion_nueva)
        
predictions_folder = "nnUNet/nnUNet_raw/Dataset022_Esclerosis/predictions/"
output_folder = "outputs_3d/"

pacientes = obtener_nombres_pacientes(source_folder)
predicciones = obtener_archivos_predicciones(predictions_folder)
copiar_y_renombrar_predicciones(pacientes, predicciones, predictions_folder, output_folder)

def limpiar_directorio(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error al borrar {file_path}. Razón: {e}')

limpiar_directorio("nnUNet/nnUNet_raw/Dataset022_Esclerosis/imagesTs/")
limpiar_directorio("nnUNet/nnUNet_raw/Dataset022_Esclerosis/predictions/")
