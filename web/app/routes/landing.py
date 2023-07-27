from app.models import check_pet
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/landing/", response_class=JSONResponse)
async def landing(request: Request, id: str):
    if check_pet(id):
        return templates.TemplateResponse(
            "landing.html", {"request": request, "id": id}
        )
    else:
        return RedirectResponse(f"/petregister/?id={id}", status_code=307)
