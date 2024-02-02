from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
from train import main

app = FastAPI(title='MLOps App')
# load the model
# predictor = ColaONNXPredictor('./models/model.onnx')


@app.get('/')
def home():
    return '<h2>This is a NLP project</h2>'


@app.get('/train')
async def train_model() -> str:
    try:
        main()
        return 'Training successful'
    except Exception as e:
        return f'Training failed:{e}'

# @app.get('/predict')
# async def get_prediction(text: str) -> dict:
#     result = predictor.predict(text)
#     return result
