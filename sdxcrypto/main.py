import uvicorn
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

#imports from this lib
from models import models
from db.db import engine
from routers import user, auth, generate
import scheduler_interface

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

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
app.include_router(auth.router)
app.include_router(generate.router)

@app.get("/")
def root():
    return {"message": "Hello World SDxCrypto"}