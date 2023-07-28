from app.models import check_id, insert_pet, insert_user
from app.utils import encrypt_password
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("sign-up.html", {"request": request})


@router.post("/register", response_class=JSONResponse)
async def register_post(
    request: Request,
    id: str = Form(...),
    password: str = Form(...),
):
    hashed_password, salt = encrypt_password(password.encode("utf-8"))
    result = insert_user(id, hashed_password.hex(), salt.hex())
    if result == "성공":
        return templates.TemplateResponse("login.html", {"request": request})
    elif result == "실패":
        return templates.TemplateResponse("sign-up.html", {"request": request})
    elif result == "ID중복":
        return templates.TemplateResponse("sign-up.html", {"request": request})


@router.get("/idcheck/", response_class=JSONResponse)
async def id_check(request: Request, id: str):
    result = check_id(id)
    if result:
        return {"result": True}
    else:
        return {"result": False}


@router.get("/petregister/", response_class=JSONResponse)
async def pet_register(request: Request, id: str):
    return templates.TemplateResponse(
        "pet-register.html", {"request": request, "id": id}
    )


@router.post("/petregister/", response_class=JSONResponse)
async def pet_register_post(
    request: Request,
    id: str,
    name: str = Form(...),
    birth: str = Form(...),
    catordog: str = Form(...),
    gender: str = Form(...),
):
    result = insert_pet(id, name, birth, catordog, gender)
    if result == "성공":
        return templates.TemplateResponse(
            "landing.html", {"request": request, "id": id}
        )
    elif result == "실패":
        return templates.TemplateResponse(
            "pet-register.html", {"request": request, "id": id}
        )
    elif result == "이미 등록된 펫":
        return templates.TemplateResponse(
            "landing.html", {"request": request, "id": id}
        )
