import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv
from segmentation_models_pytorch import MAnet
from synology_api import filestation

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
        A.PadIfNeeded(416, 416, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ]
)

app = Celery(
    "worker_app", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)

logger = get_task_logger(__name__)

app.conf.task_routes = {
    "cat_skin": {"queue": "cat_skin"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("../checkpoint/manetmit.ckpt", map_location=device)
state_dict = {k.partition("model.")[2]: v for k, v in ckpt["state_dict"].items()}

model = MAnet(
    encoder_name="mit_b5",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
model.load_state_dict(state_dict)
model.eval()
model.to(device=device)


def plot_cat_skin_seg(img, pred):
    pred = pred.squeeze()
    pred = torch.sigmoid(pred).detach().cpu()

    pred_thresh = pred > 0.5

    chan, height, width = pred.shape

    colors = [(138, 43, 226), (188, 238, 104), (255, 215, 0)]

    num_cont = np.zeros(chan)
    height_cont = np.zeros(chan)
    width_cont = np.zeros(chan)

    for c in range(chan):
        pred_thresh_c = pred_thresh[c]
        coords_c = np.stack(np.where(pred_thresh_c), axis=1)
        num_coords_c = coords_c.shape[0]

        if num_coords_c > 0:
            num_cont[c] = num_coords_c
            height_cont[c] = np.sum(coords_c[:, 0])
            width_cont[c] = np.sum(coords_c[:, 1])

    idx = np.argmax(num_cont)
    if num_cont[idx] > 100:
        height = int(height_cont[idx] / num_cont[idx])
        width = int(width_cont[idx] / num_cont[idx])
        img = cv2.circle(img, (width, height), 40, colors[idx], 3)

    return img, idx


@app.task(name="cat_skin")
def cat_skin(image_path):
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
    output = model.predict(x)
    result, idx = plot_cat_skin_seg(img, output)
    cv2.imwrite("./temp/" + image_path + "_seg.jpg", result)

    fl.upload_file(
        dest_path="/BoostCamp/Result_Image",
        file_path="./temp/" + image_path + "_seg.jpg",
    )

    return {"image_path": image_path, "output": str(idx)}
