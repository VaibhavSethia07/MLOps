FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]