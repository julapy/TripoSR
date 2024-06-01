import logging
import os
import time

import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"

timer.start("Initializing model")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
timer.end("Initializing model")

timer.start("Processing image")

rembg_session = rembg.new_session()

image_path = "examples/chair.png"
image = remove_background(Image.open(image_path), rembg_session)
image = resize_foreground(image, 0.85)
image = np.array(image).astype(np.float32) / 255.0
image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
image = Image.fromarray((image * 255.0).astype(np.uint8))
image.save(os.path.join(output_dir, "input.png"))

timer.end("Processing image")

timer.start("Running model")
with torch.no_grad():
    scene_codes = model([image], device=device)
timer.end("Running model")

timer.start("Extracting mesh")
meshes = model.extract_mesh(scene_codes, True, resolution=256)
timer.end("Extracting mesh")

out_mesh_path = os.path.join(output_dir, "mesh.obj")
timer.start("Exporting mesh")
meshes[0].export(out_mesh_path)
timer.end("Exporting mesh")