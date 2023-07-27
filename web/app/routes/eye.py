import os
import shutil
from uuid import uuid4

import cv2
from app.models import get_pet, get_script
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

dog_classes = ["유루증", "안검내반증", "안검염", "결막염", "안검종양"]
cat_classes = ["결막염", "안검염", "각막궤양", "각막부골편", "비궤양성각막염"]

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/eye/", response_class=JSONResponse)
async def eye(request: Request, id: str):
    return templates.TemplateResponse("model3.html", {"request": request, "id": id})


@router.post("/eye_predict", response_class=JSONResponse)
async def eye_predict(image: UploadFile = File(...), id: str = Form(...)):
    UPLOAD_DIRECTORY = "temp"
    os.makedirs("temp", exist_ok=True)
    os.makedirs("app/static/result", exist_ok=True)
    pet = get_pet(id)
    print(pet)
    try:
        image_name = f"{str(uuid4())}"
        image_path = os.path.join(UPLOAD_DIRECTORY, image_name + ".jpg")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=image_path)

        result = celery_obj.send_task("eyes_crop", args=[image_name], queue="eyes_crop")
        crop_info = result.get()

        print(crop_info)
        img = cv2.imread(image_path)
        print(img.shape)
        img = img[
            int(crop_info["y_bottom_left"]) : int(crop_info["y_upper_right"]),
            int(crop_info["x_bottom_left"]) : int(crop_info["x_upper_right"]),
            ...,
        ]
        print(f"{img.shape=}")
        cv2.imwrite(image_path, img)

        fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=image_path)

        os.remove(image_path)
        if pet.cat_dog == "dog":
            result = celery_obj.send_task(
                "dog_eyes", args=[image_name], queue="dog_eyes"
            )
        elif pet.cat_dog == "cat":
            result = celery_obj.send_task(
                "cat_eyes", args=[image_name], queue="cat_eyes"
            )

        response = result.get()
        fl.get_file(
            path="/BoostCamp/Result_Image/" + image_name + "_gradcam.jpg",
            mode="download",
            dest_path="./app/static/result",
        )

        if pet.cat_dog == "dog":
            response["desease"] = dog_classes
            response["name"] = pet.name
            response["maxDesease"] = dog_classes[
                response["output"][0].index(max(response["output"][0]))
            ]
            script = get_script(pet.cat_dog, "eye", response["maxDesease"])
            response["symptom"] = script.symptom
            response["cause"] = script.cause
            response["action"] = script.action
        elif pet.cat_dog == "cat":
            response["desease"] = cat_classes
            response["name"] = pet.name
            response["maxDesease"] = cat_classes[
                response["output"][0].index(max(response["output"][0]))
            ]
            script = get_script(pet.cat_dog, "eye", response["maxDesease"])
            response["symptom"] = script.symptom
            response["cause"] = script.cause
            response["action"] = script.action

        return JSONResponse(response)
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": str(e)}, status_code=500)


@router.post("/test", response_class=JSONResponse)
async def test(cat_dog: str, part: str, desease: str):
    result = get_script(cat_dog, part, desease)
    return {"symptom": result.symptom, "cause": result.cause, "action": result.action}
