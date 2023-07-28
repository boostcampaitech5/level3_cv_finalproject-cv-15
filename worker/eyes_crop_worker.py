import os
import torch
import mmcv

from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv
from synology_api import filestation

from mmdet.apis import init_detector, inference_detector

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

app = Celery(
    "worker_app", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)

logger = get_task_logger(__name__)

app.conf.task_routes = {
    "eyes_crop": {"queue": "eyes_crop"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
config_file = '../src/config/mmdet_config/train_config/yolov3_eye_detection.py'
checkpoint_file = '../checkpoint/eye_detection.pth'
model = init_detector(config_file, checkpoint_file, device=device)

@app.task(name="eyes_crop")
def eyes_crop(image_path):
    logger.info(f"input image path : {image_path}")
    fl.get_file(
        path="/BoostCamp/Inference_Image/" + image_path + ".jpg",
        mode="download",
        dest_path="./temp",
    )
    image = mmcv.imread("./temp/" + image_path + ".jpg", channel_order='rgb')
    output_dict = inference_detector(model, image)

    bboxes = output_dict.pred_instances.bboxes
    x_bottom_left = str(int(bboxes[0][0].cpu().numpy()))
    y_bottom_left = str(int(bboxes[0][1].cpu().numpy()))
    x_upper_right = str(int(bboxes[0][2].cpu().numpy()))
    y_upper_right = str(int(bboxes[0][3].cpu().numpy()))

    fl.upload_file(
        dest_path="/BoostCamp/Inference_Image",
        file_path="./temp/" + image_path + ".jpg",
    )

    return {"image_path": image_path, "x_bottom_left": x_bottom_left, "y_bottom_left": y_bottom_left,
            "x_upper_right": x_upper_right, "y_upper_right": y_upper_right} 
