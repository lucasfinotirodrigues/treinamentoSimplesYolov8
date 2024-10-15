from ultralytics import YOLO

# Carregue o modelo YOLO localmente
model = YOLO('yolov8l-seg.pt')  # Carregue o caminho correto do modelo local

# Execute a previsão (segmentação)
results = model.predict(source='horse.jpg', conf=0.25, save=True)

# Exiba a imagem resultante no ambiente local
from PIL import Image
img = Image.open('horse.jpg')
