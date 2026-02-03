import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Configuración
input_folder = 'dataset' # Carpeta raíz de tus datos
output_dir = 'trajectory_images' # Carpeta donde se guardarán las imágenes

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Buscar todos los archivos CSV recursivamente
csv_files = glob.glob(os.path.join(input_folder, '**/*.csv'), recursive=True)
print(f"Se encontraron {len(csv_files)} archivos CSV.")

# Características a graficar
features = ['Surge Velocity', 'Sway Velocity', 'Yaw Rate', 'Yaw Angle']

for file_path in csv_files:
    try:
        # Leer el dataset
        df = pd.read_csv(file_path)
        
        # Obtener el nombre del archivo para usarlo en el guardado (sin .csv)
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Crear figura con 4 subplots (4 filas, 1 columna)
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Trajectory Features: {name_no_ext}', fontsize=16)
        
        # Graficar cada feature
        for i, feature in enumerate(features):
            if feature in df.columns:
                axes[i].plot(df['Time (s)'], df[feature], label=feature)
                axes[i].set_ylabel(feature)
                axes[i].grid(True, linestyle='--', alpha=0.7)
                axes[i].legend(loc='upper right')
            else:
                axes[i].text(0.5, 0.5, f'{feature} no encontrado', 
                             ha='center', va='center', transform=axes[i].transAxes)
        
        axes[-1].set_xlabel('Time (s)')
        
        # Ajustar diseño y guardar imagen
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'{name_no_ext}.png')
        plt.savefig(save_path)
        plt.close(fig) # Cerrar para liberar memoria
        
        print(f"Guardado: {save_path}")

    except Exception as e:
        print(f"Error procesando {file_path}: {e}")

print("Proceso completado.")