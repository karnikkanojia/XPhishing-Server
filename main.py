from enum import Enum
from fastapi import FastAPI

app = FastAPI()

class Model(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/models/{model_name}")
async def get_model(model_name: Model):
    if model_name is Model.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    print(type(file_path))
    return {"file_path": file_path}

@app.get("/find")
async def find(q: str = None):
    results = {"inference": 0}
    return results