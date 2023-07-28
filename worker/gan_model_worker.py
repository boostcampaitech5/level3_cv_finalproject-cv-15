import os

import requests
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv
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

app = Celery(
    "worker_app", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)

logger = get_task_logger(__name__)

app.conf.task_routes = {
    "gan_model": {"queue": "gan_model"},
}


@app.task(name="gan_model")
def gan_model(image_path):
    logger.info(f"input image path : {image_path}")
    fl.get_file(
        path="/BoostCamp/Inference_Image/" + image_path + ".jpg",
        mode="download",
        dest_path="../temp",
    )

    response = requests.get(
        f"http://localhost:8000/gan_predict/?image_path={image_path}"
    )
    print(response.content)

    fl.upload_file(
        dest_path="/BoostCamp/Result_Image",
        file_path="../temp/" + image_path + "_transfer.jpg",
    )

    return {"image_path": image_path}
