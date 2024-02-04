FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://1r8ooyAQuFeA2t78aMALTb28oCOKJuPRK
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path ../creds.json

# pulling the trained model
RUN dvc pull models/best-checkpoint.ckpt.dvc models/model.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]