FROM huggingface/accelerate-gpu

WORKDIR /project

COPY ./requirements.txt .

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/huggingface/diffusers.git  

#set while building
ARG hf_token
ENV HF_ACCESS_TOKEN=$var_name

COPY . .

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]


