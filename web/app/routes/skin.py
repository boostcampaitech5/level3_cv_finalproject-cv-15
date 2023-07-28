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

dog_skin_class = ["구진/플라크", "비듬/각질/상피성잔고리", "태선화/과다색소침착", "농포/여드름", "미란/궤양", "결절/종괴"]
dog_query_key = {
    "구진/플라크": ["구진", "플라크"],
    "비듬/각질/상피성잔고리": ["비듬/각질", "상피성잔고리"],
    "태선화/과다색소침착": ["태선화", "과다색소침착"],
    "농포/여드름": ["농포", "여드름"],
    "미란/궤양": ["미란", "궤양"],
    "결절/종괴": ["결절", "종괴"],
}
cat_skin_class = ["비듬/각질/상피성잔고리", "농포/여드름", "결절/종괴"]
cat_query_key = {
    "비듬/각질/상피성잔고리": ["비듬/각질", "상피성잔고리"],
    "농포/여드름": ["농포", "여드름"],
    "결절/종괴": ["결절", "종괴"],
}

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/skin/", response_class=JSONResponse)
async def skin(request: Request, id: str):
    return templates.TemplateResponse("model1.html", {"request": request, "id": id})


@router.post("/skin_predict", response_class=JSONResponse)
async def skin_predict(image: UploadFile = File(...), id: str = Form(...)):
    UPLOAD_DIRECTORY = "temp"
    os.makedirs("temp", exist_ok=True)
    os.makedirs("app/static/result", exist_ok=True)
    pet = get_pet(id)
    try:
        image_name = f"{str(uuid4())}"
        image_path = os.path.join(UPLOAD_DIRECTORY, image_name + ".jpg")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=image_path)

        os.remove(image_path)
        if pet.cat_dog == "dog":
            result = celery_obj.send_task(
                "dog_skin", args=[image_name], queue="dog_skin"
            )
        elif pet.cat_dog == "cat":
            result = celery_obj.send_task(
                "cat_skin", args=[image_name], queue="cat_skin"
            )

        response = result.get()

        if pet.cat_dog == "dog":
            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_det.jpg",
                mode="download",
                dest_path="./app/static/result",
            )

            img = cv2.imread("./app/static/result/" + image_name + "_det.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./app/static/result/" + image_name + "_det.jpg", img)
        elif pet.cat_dog == "cat":
            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_seg.jpg",
                mode="download",
                dest_path="./app/static/result",
            )

            img = cv2.imread("./app/static/result/" + image_name + "_seg.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./app/static/result/" + image_name + "_seg.jpg", img)

        if pet.cat_dog == "dog":
            response["desease"] = dog_skin_class
            response["name"] = pet.name
            response["maxDesease"] = dog_skin_class[int(response["output"])]

            symptom = []
            cause = []
            action = []
            deseaseList = []
            for key in dog_query_key[response["maxDesease"]]:
                script = get_script(pet.cat_dog, "skin", key)
                symptom.append(script.symptom)
                cause.append(script.cause)
                action.append(script.action)
                deseaseList.append(script.desease)
            response["symptom"] = symptom
            response["cause"] = cause
            response["action"] = action
            response["deseaseList"] = deseaseList
            response["image_path"] = response["image_path"] + "_det"
        elif pet.cat_dog == "cat":
            print(response["output"])
            response["desease"] = cat_skin_class
            response["name"] = pet.name
            response["maxDesease"] = cat_skin_class[int(response["output"])]

            symptom = []
            cause = []
            action = []
            deseaseList = []
            for key in cat_query_key[response["maxDesease"]]:
                script = get_script(pet.cat_dog, "skin", key)
                symptom.append(script.symptom)
                cause.append(script.cause)
                action.append(script.action)
                deseaseList.append(script.desease)
            response["symptom"] = symptom
            response["cause"] = cause
            response["action"] = action
            response["deseaseList"] = deseaseList
            response["image_path"] = response["image_path"] + "_seg"

        return JSONResponse(response)
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": str(e)}, status_code=500)
