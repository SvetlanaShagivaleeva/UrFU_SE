from io import BytesIO

from PIL import Image
import torch
import cv2
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


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
    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload = False)

        img = Image.open(BytesIO(await file.read()))
        results = model(img)

        json_results = results_to_json(results,model)
        return JSONResponse({"data": json_results,
                                "message": "object detected successfully",
                                "errors": None},
                            status_code=200)
    except Exception as error:
            return JSONResponse({"message": "object detection failed",
                                 "errors": "error"},
                                status_code=400)


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