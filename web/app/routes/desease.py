from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/disease/", response_class=JSONResponse)
async def disease(request: Request, id: str):
    return templates.TemplateResponse("disease.html", {"request": request, "id": id})


@router.get("/dog_eye_disease/", response_class=JSONResponse)
async def dog_eye_disease(request: Request, id: str):
    return templates.TemplateResponse(
        "disease-dog-eye.html", {"request": request, "id": id}
    )


@router.get("/dog_skin_disease/", response_class=JSONResponse)
async def dog_skin_disease(request: Request, id: str):
    return templates.TemplateResponse(
        "disease-dog-skin.html", {"request": request, "id": id}
    )


@router.get("/cat_eye_disease/", response_class=JSONResponse)
async def cat_eye_disease(request: Request, id: str):
    return templates.TemplateResponse(
        "disease-cat-eye.html", {"request": request, "id": id}
    )


@router.get("/cat_skin_disease/", response_class=JSONResponse)
async def cat_skin_disease(request: Request, id: str):
    return templates.TemplateResponse(
        "disease-cat-skin.html", {"request": request, "id": id}
    )
