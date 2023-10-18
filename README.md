# UrFU_SE
Выполнение практических заданий по программной инженерии
## Участники команды
- Баев Артем Александрович (РИМ-130906)
- Шагивалеева Светлана Ринатовна (РИМ-130906)
## Описание модели
С помощью модели yolov5 можно детектировать до 80-ти классов объектов на изображении [(список классов)](https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/classes.yaml).
В качестве входных данных принимается изображение, а выходные данные это список детекций с семью переменныеми: xmin, ymin, xmax, ymax, confidence, class, name. Пример отрисовки детекции представлен на [изображении](https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/runs/detect/exp/zidane.jpg).
## Использование модели
```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```
