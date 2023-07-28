import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv
from synology_api import filestation
from timm import create_model
from torch.nn.functional import sigmoid

from utils import apply_heatmap, get_gradcam

load_dotenv()

fl = filestation.FileStation(
    ip_address=os.environ["NAS_ADDR"],
    port=os.environ["NAS_PORT"],
    username=os.environ["NAS_ID"],
    password=os.environ["NAS_PASS"],
    secure=False,
    cert_verify=False,
    dsm_version=7,
)

transform = A.Compose(
    [
        # A.Sharpen(p=1.0, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        # A.clahe(p=1.0, clip_limit=(1, 4), tile_grid_size=(8, 8)),
        A.Resize(400, 400),
        A.Normalize(),
        ToTensorV2(),
    ]
)

app = Celery(
    "worker_app", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)

logger = get_task_logger(__name__)

app.conf.task_routes = {
    "cat_eyes": {"queue": "cat_eyes"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("../checkpoint/cat_eye_resnest50.ckpt", map_location=device)
state_dict = {k.partition("model.")[2]: v for k, v in ckpt["state_dict"].items()}

model = create_model(model_name="resnest50d.in1k", pretrained=True, num_classes=5)
model.load_state_dict(state_dict)
model.eval()
model.to(device=device)


@app.task(name="cat_eyes")
def cat_eyes(image_path):
    logger.info(f"input image path : {image_path}")
    fl.get_file(
        path="/BoostCamp/Inference_Image/" + image_path + ".jpg",
        mode="download",
        dest_path="./temp",
    )

    img = cv2.imread("./temp/" + image_path + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(image=img)["image"].unsqueeze(0)
    x = x.to(device=device)
    heatmap, output = get_gradcam(model, x)
    result = apply_heatmap(img, heatmap.detach().cpu().numpy())
    cv2.imwrite("./temp/" + image_path + "_gradcam.jpg", result)
    fl.upload_file(
        dest_path="/BoostCamp/Result_Image",
        file_path="./temp/" + image_path + "_gradcam.jpg",
    )
    os.remove("./temp/" + image_path + "_gradcam.jpg")
    os.remove("./temp/" + image_path + ".jpg")
    output = sigmoid(output)
    return {"image_path": image_path, "output": output.detach().cpu().numpy().tolist()}
