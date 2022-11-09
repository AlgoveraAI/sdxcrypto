import os
import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method
# from fastapi.middleware.cors import CORSMiddleware

#imports from this lib
from utils import createLogHandler
from routers import user, generate, assets
import scheduler_interface
from manual_setup import initial_setup

#CREATE LOGGER
logger = createLogHandler(__name__, 'logs.log')

#EXTEND APP
app = FastAPI()

#START SCHEDULER
set_start_method('spawn')
scheduler_interface.init_scheduler()

# origins = ["*"]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

#SETUP ROUTES
app.include_router(user.router)
app.include_router(generate.router)
app.include_router(assets.router)

@app.get("/")
def root():
    return {"message": "Hello World SDxCrypto"}


