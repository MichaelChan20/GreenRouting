FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Python 3.11
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
       python3.11 python3.11-venv python3.11-dev python3-pip \
       git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
RUN mkdir -p /app/results
WORKDIR /app

# Requirements
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt


# Run eval
ENV PYTHONUNBUFFERED=1
# ENV HF_ALLOW_CODE_EVAL=1

COPY 3-model-energy.py .
ENTRYPOINT ["python3.11", "-u", "3-model-energy.py"]
