FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

ARG UID=1000
ARG GID=1000
ARG USERNAME=shikina
ARG GROUPNAME=shikina

WORKDIR /opt/app

RUN apt update && apt upgrade -y && apt install -y wget build-essential pixz python3 graphviz python3-pip libboost-python-dev
RUN apt autoremove -y

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
RUN mkdir /opt/app/data
RUN chown -R $USERNAME:$GROUPNAME /opt/app
USER $USERNAME
RUN pip install neat-python bson graphviz matplotlib numpy numba

COPY scheduler.py /opt/app/
COPY config/ /opt/app/config/

ENTRYPOINT ["python3", "scheduler.py"]
