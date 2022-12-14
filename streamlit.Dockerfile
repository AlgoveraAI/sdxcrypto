FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -e .

#set while building
ARG hf_token
ENV HF_ACCESS_TOKEN=$hf_token
RUN export HF_ACCESS_TOKEN=$var_name

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "sdxcrypto/app.py", "--server.port=8501", "--server.address=0.0.0.0"]