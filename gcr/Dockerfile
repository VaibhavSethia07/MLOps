FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG CREDENTIAL_FILE_PATH

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y
    
RUN gcloud auth activate-service-account --key-file=$CREDENTIAL_FILE_PATH

# install requirements
RUN pip install "dvc[gcs]"   # since gcs is the remote storage
RUN pip install "dvc[gs]"
RUN pip install -r requirements.txt

# initialise dvc
RUN dvc init --no-scm

# configuring remote server in dvc
RUN dvc remote add -d model-store gs://models-dvc-04-02-2024/trained-models
RUN dvc remote modify model-store projectname mlops-413215
RUN dvc remote modify --local model-store credentialpath $CREDENTIAL_FILE_PATH

RUN cat .dvc/config

# pulling the trained model
RUN dvc pull models/best-checkpoint.ckpt.dvc models/model.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]