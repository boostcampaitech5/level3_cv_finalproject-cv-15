import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from synology_api import filestation
from timm import create_model

from celery import Celery
from celery.utils.log import get_task_logger
from src.utils import apply_heatmap, get_gradcam

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
        A.LongestMaxSize(400),
        A.PadIfNeeded(400, 400, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ]
)

app = Celery(
    "worker_app", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)

logger = get_task_logger(__name__)

app.conf.task_routes = {
    "dog_eyes": {"queue": "dog_eyes"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("../checkpoint/dog_eyes.ckpt", map_location=device)
state_dict = {k.partition("model.")[2]: v for k, v in ckpt["state_dict"].items()}

model = create_model(model_name="resnest50d.in1k", pretrained=True, num_classes=5)
model.load_state_dict(state_dict)
model.eval()


@app.task(name="dog_eyes")
def dog_eyes(image_path):
    logger.info(f"input image path : {image_path}")
    fl.get_file(
        path="/BoostCamp/Inference_Image/" + image_path + ".jpg",
        mode="download",
        dest_path="./temp",
    )

    img = cv2.imread("./temp/" + image_path + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(image=img)["image"]

    for i in range(5):
        heatmap, output = get_gradcam(img.to(device=device))
        result = apply_heatmap(img, heatmap.numpy())
        cv2.imwrite("./temp/" + image_path + f"_{i}" + ".jpg", result)
        fl.upload_file(
            dest_path="/BoostCamp/Result_Image",
            file_path="./temp/" + image_path + f"_{i}" + ".jpg",
        )
        os.remove("./temp/" + image_path + f"_{i}" + ".jpg")
    os.remove("./temp/" + image_path + ".jpg")

    return {"image_path": image_path, "output": output.detach().cpu().numpy()}
