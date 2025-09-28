# Use the official Ubuntu image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
env TZ=Etc/UTC

# Update and install basic stuff
RUN apt-get update && apt-get install -y \
    curl \
    nano \
    python3 \
    pip 

RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 torchtext==0.15.2 datasets==2.15.0 tokenizers==0.13.3 torchmetrics==1.0.3 \
    tensorboard==2.13.0 altair==5.1.1 wandb==0.15.9 jupyter

# Install additional dependencies
RUN pip3 install "jinja2<3.1.0"

# Set working directory
WORKDIR /workspace

COPY model.py /workspace/model.py
COPY main.ipynb /workspace/main.ipynb

# Expose port for Jupyter
EXPOSE 8888

# Run Jupyter when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

