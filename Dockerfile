FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && \
    apt-get install -y \
        libglib2.0-0 \
        libgl1-mesa-glx \
        git \
        vim \
        g++ \
        sudo \
        xvfb \
        libxrender-dev \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chmod -R 777 /opt/conda

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN pip install \
    opencv-python \
    timm==1.0.15 \
    xformers==0.0.29 \
    'diffusers[torch]==0.34.0' \
    tensorboard==2.19.0 \
    einops==0.8.1 \
    transformers==4.48.3 \
    scikit-image==0.25.2 \
    omegaconf==2.3.0 \
    torchdiffeq==0.2.5 \
    'imageio[ffmpeg]'

RUN sudo chmod -R 777 /opt/conda
WORKDIR /workspaces/AngioFILM
