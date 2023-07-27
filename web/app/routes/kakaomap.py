from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/kakao_map/", response_class=JSONResponse)
async def kakaomap(request: Request, id: str):
    return templates.TemplateResponse("kakao_map.html", {"request": request, "id": id})
