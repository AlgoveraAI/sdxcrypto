import os
import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method
# from fastapi.middleware.cors import CORSMiddleware

#imports from this lib
from utils import createLogHandler
from routers import user, generate, assets
import scheduler_interface

#Create logger
logger = createLogHandler(__name__, 'logs.log')

#Load env variables
app = FastAPI()

#start scheduler
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

app.include_router(user.router)
app.include_router(generate.router)
app.include_router(assets.router)

@app.get("/")
def root():
    return {"message": "Hello World SDxCrypto"}