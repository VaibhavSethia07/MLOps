from typing import List

from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from train import main

provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

summary = '''This is an application powered by NLP. 
This app can classify a sentence into one of the two classes.
❌ Unacceptable: Grammatically not correct
✅ Acceptable: Grammatically correct
'''
app = FastAPI(title='MLOps App', summary=summary)
# load the model
predictor = ColaONNXPredictor('./models/model.onnx')


@app.get('/')
def home():
    return summary


@app.get('/predict')
async def get_prediction(text: str):
    with tracer.start_as_current_span('pytorch-lightning-span') as span:
        result = predictor.predict(text)

        return result
