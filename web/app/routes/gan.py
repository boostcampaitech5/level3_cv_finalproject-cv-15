import os
import shutil
from uuid import uuid4

from celery import Celery
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
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

celery_obj = Celery(
    "tasks", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BACKEND"]
)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/gan/", response_class=JSONResponse)
async def gan(request: Request, id: str):
    return templates.TemplateResponse("model5.html", {"request": request, "id": id})


@router.post("/gan_predict", response_class=JSONResponse)
async def gan_predict(image: UploadFile = File(...), id: str = Form(...)):
    UPLOAD_DIRECTORY = "temp"
    os.makedirs("temp", exist_ok=True)
    os.makedirs("app/static/result", exist_ok=True)
    try:
        image_name = f"{str(uuid4())}"
        image_path = os.path.join(UPLOAD_DIRECTORY, image_name + ".jpg")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=image_path)

        os.remove(image_path)

        result = celery_obj.send_task("gan_model", args=[image_name], queue="gan_model")

        response = result.get()

        fl.get_file(
            path="/BoostCamp/Result_Image/" + image_name + "_transfer.jpg",
            mode="download",
            dest_path="./app/static/result",
        )

        return JSONResponse(response)
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
