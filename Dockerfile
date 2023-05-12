FROM nvcr.io/nvidia/dli/dli-nano-ai:v2.0.2-r32.7.1ja

# cupy
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 42D5A192B819C5DA
RUN apt update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install cupy-cuda102 -f https://pip.cupy.dev/aarch64

# discord.py
RUN python3.8 -m pip install -U "discord.py==1.7.3"

# mecab, gensim
RUN python3.8 -m pip install mecab-python3 unidic-lite gensim

RUN mkdir /dir
WORKDIR /dir
VOLUME ["/dir"]

CMD ["bash"]
