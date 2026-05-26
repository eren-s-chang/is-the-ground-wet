FROM python:3.12-slim
WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libxext6 libxrender1 libsm6 libxcb1 curl git \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -o /app/sam_vit_b_01ec64.pth
COPY app.py ./
EXPOSE 9000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
