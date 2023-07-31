import os
import cv2
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
    "dog_skin": {"queue": "dog_skin"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
config_file = '../src/config/mmdet_config/train_config/cascade-rcnn_convnext_dog_skin.py'
checkpoint_file = '../checkpoint/dog_skin_convnext_linearLr.pth'
model = init_detector(config_file, checkpoint_file, device=device)

def plot_dog_skin_det(img, output_dict):
    bboxes = output_dict.pred_instances.bboxes.cpu()
    labels = output_dict.pred_instances.labels.cpu()
    scores = output_dict.pred_instances.scores.cpu()
    print('labels = ', labels)

    colors = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),]

    idxs = []
    score_list = []
    for i, bbox in enumerate(bboxes):
        print(bbox)
        height = int((bbox[1] + bbox[3]) / 2)
        width = int((bbox[0] + bbox[2]) / 2)
        r = int(max((bbox[3]-bbox[1]), bbox[2]-bbox[0]) / 2)
        idx = int(labels[i])
        idxs.append(idx)
        score_list.append(float(scores[i]))
        img = cv2.circle(img, (width, height), r, colors[idx], 3)

    return img, idxs, score_list


@app.task(name="dog_skin")
def dog_skin(image_path):
    logger.info(f"input image path : {image_path}")
    fl.get_file(
        path="/BoostCamp/Inference_Image/" + image_path + ".jpg",
        mode="download",
        dest_path="./temp",
    )
    image = mmcv.imread("./temp/" + image_path + ".jpg", channel_order='rgb')
    output_dict = inference_detector(model, image)
    result, idxs, score_list = plot_dog_skin_det(image, output_dict)
    print(idxs)
    cv2.imwrite("./temp/" + image_path + "_det.jpg", result)

    fl.upload_file(
        dest_path="/BoostCamp/Result_Image",
        file_path="./temp/" + image_path + "_det.jpg",
    )

    return {"image_path": image_path, "output": str(idxs[0]) , "scores": score_list}
