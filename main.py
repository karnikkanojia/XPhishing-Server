from enum import Enum
from fastapi import FastAPI
from utils.predict import predict
app = FastAPI()

@app.get("/find")
def find(url: str = None):
    print(url)
    results = predict(url)
    print(results)
    if 'Benign' in results:
        print('here')
        return {'isMalicious': 0}
    print('there')
    return {'isMalicious': 1}