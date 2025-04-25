import os
import shutil
from sklearn.model_selection import train_test_split

# Configurações
dataset_path = "C:/Users/katys/OneDrive/Documentos/GitHub/AgroIntelliVision/data/DataSet"
train_ratio = 0.8  # 80% treino, 20% validação

# Cria pastas de treino/validação
os.makedirs(os.path.join(dataset_path, "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val"), exist_ok=True)

# Para cada classe
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path) and class_name not in ["train", "val"]:
        # Lista imagens
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png', '.jpeg'))]  # Adicione mais formatos aqui
        
        # Divide em treino/validação
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Cria subpastas e copia arquivos
        for folder, imgs in [("train", train_imgs), ("val", val_imgs)]:
            os.makedirs(os.path.join(dataset_path, folder, class_name), exist_ok=True)
            for img in imgs:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(dataset_path, folder, class_name, img)
                )

print("Dataset dividido com sucesso!")