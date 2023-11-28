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
## Использование streamlit
Для запуска веб интерфейса необходимо ввести следующую команду в консоли:
```bash
streamlit run streamlit_demo.py
```
В браузере откроется окно с элементом для загрузки изображения:
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/streamlit_demo_image1.jpg"></a>
</div>
После загрузки изображения оно будет отображено под элементом загрузки изображения:
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/streamlit_demo_image2.jpg"></a>
</div>
При загрузки изображения и нажатии кнопки "Задетектить изображение" появится изображения с обнаруженными объектами. Объекты будут выделены красным прямоугольником, над которым будет написан класс объекта и уверенность в нем:
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/streamlit_demo_image3.jpg"></a>
</div>

## Использование FastAPI
Для запуска сервера необходимо ввести следующую команду в консоли:
```bash
python server.py
```
Заходим в браубере по адресу http://localhost:8000/
Откроется окно с интерфейсом сайта
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/fastapi_demo_image1.jpg"></a>
</div>
В данном окне можно загрузить изображения для детекций, нажав кнопку "выбрать файл". А также одну из моделей yolov5
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/fastapi_demo_image2.jpg"></a>
</div>
Нажав кнопку "Submit" получим изображение с задетектированными объектами
<div align="center">
  <img width="100%" src="https://github.com/SvetlanaShagivaleeva/UrFU_SE/blob/main/data/fastapi_demo_image3.jpg"></a>
</div>
