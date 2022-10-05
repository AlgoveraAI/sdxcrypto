FROM nvcr.io/nvidia/pytorch:21.09-py3 

WORKDIR /app

COPY . .

RUN pip install -e .

EXPOSE 5000

CMD ["uvicorn", "sdxcrypto.main:app", "--reload", "--host", "0.0.0.0", "--port", "5000"]
