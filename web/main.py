from app.routes.desease import router as desease_router
from app.routes.eye import router as eye_router
from app.routes.gan import router as gan_router
from app.routes.kakaomap import router as map_router
from app.routes.landing import router as landing_router
from app.routes.login import router as login_router
from app.routes.register import router as register_router
from app.routes.skin import router as skin_router
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(login_router)
app.include_router(register_router)
app.include_router(landing_router)
app.include_router(skin_router)
app.include_router(eye_router)
app.include_router(gan_router)
app.include_router(map_router)
app.include_router(desease_router)
