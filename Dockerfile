FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir kaggle pandas numpy scikit-learn

RUN pip3 install --no-cache-dir joblib

CMD ["/bin/bash", "-lc", "python3 train_nn.py && python3 predict_nn.py"]