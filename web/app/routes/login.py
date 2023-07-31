from app.models import get_user
from app.utils import check_password
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    id: str = Form(...),
    password: str = Form(...),
):
    result = get_user(id)

    if check_password(
        password.encode("utf-8"), bytes.fromhex(result.pw), bytes.fromhex(result.salt)
    ):
        return RedirectResponse(f"/landing/?id={id}", status_code=302)
    else:
        return templates.TemplateResponse("login.html", {"request": request})
