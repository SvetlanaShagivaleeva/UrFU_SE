from io import BytesIO

from PIL import Image
import torch
import cv2
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
from streamlit_demo import int_to_classes


app = FastAPI()


@app.get("/")
async def home_page():
    html_content = '''
          <form method="post" enctype="multipart/form-data">
          <div>
              <label>Upload Image</label>
              <input name="file" type="file" multiple>
              <div>
              <label>Select YOLO Model</label>
              <select name="model_name">
                  <option>yolov5s</option>
                  <option>yolov5m</option>
                  <option>yolov5l</option>
                  <option>yolov5x</option>
              </select>
              </div>
          </div>
          <button type="submit">Submit</button>
          </form>
    '''

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/")
async def processing_request(
    file: UploadFile = File(...), 
    model_name: str = Form(...)
):
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload = False)

    img = Image.open(BytesIO(await file.read()))
    results = model(img)
    img = show_prediction(img, results)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, im_png = cv2.imencode(".png", img)
    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/png")

    # Return result in json
    # json_results = results_to_json(results,model)
    # return json_results


def show_prediction(img, preds):
  img = np.array(img)
  for data in preds.xyxy[0]:
      x0, y0, xk, yk, score, id_class = data
      x0, y0, xk, yk, id_class = int(x0), int(y0), int(xk), int(yk), int(id_class)
      img = cv2.rectangle(img, (x0, y0), (xk, yk), (255, 0, 0), 2) 
      img = cv2.putText(img, f'{int_to_classes[id_class + 1]}', (x0 + 90, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)
      img = cv2.putText(img, f'{score:.2}:', (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)
  
  return img


def results_to_json(results, model):
    data = []
    for result in results.xyxy:
        for pred in result:
            data.append({
            "class": int(pred[5]),
            "class_name": model.model.names[int(pred[5])],
            "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
            "confidence": float(pred[4]),
            })
    return data
    

if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)