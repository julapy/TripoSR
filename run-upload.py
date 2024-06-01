from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

import io
import os
import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model
    global rembg_session
    print("Loading model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to("cuda:0")
    rembg_session = rembg.new_session()
    print("Model loaded.")

@app.post("/mesh/")
async def segment(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image_data = await file.read()
        image = remove_background(Image.open(io.BytesIO(image_data)), rembg_session)
        image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        image.save("output/input.png")

        with torch.no_grad():
            scene_codes = model([image], device="cuda:0")
        meshes = model.extract_mesh(scene_codes, True, resolution=256)

        obj_stream = io.BytesIO()
        meshes[0].export(obj_stream, file_type="obj")
        obj_data = obj_stream.getvalue().decode('utf-8')

        return PlainTextResponse(content=obj_data, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


