import os

from celery import Celery
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()
celery_obj = Celery(
    "tasks", broker=os.environ["REDIS_BROKER"], backend=os.environ["REDIS_BROKER"]
)


@app.get("/dog_eye_predict/")
async def predict(path: str):
    result = celery_obj.send_task("dog_eyes", args=[path], queue="dog_eyes")

    return result.get()
