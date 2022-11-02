FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

ARG UID=1000
ARG GID=1000
ARG USERNAME=shikina
ARG GROUPNAME=shikina

WORKDIR /opt/app

RUN apt update && apt upgrade -y && apt install pixz firefox-geckodriver python3-pip python3-opengl -y

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

COPY analyze.ipynb /opt/app/
RUN chown -R $USERNAME:$GROUPNAME /opt/app

USER $USERNAME

RUN pip install -U pip
RUN pip install numpy pandas notebook ipywidgets bokeh bson selenium networkx scipy glfw PyOpenGL

RUN mkdir /opt/app/analysis

RUN mkdir /home/shikina/.jupyter/
COPY .jupyter/ /home/shikina/.jupyter/
ENTRYPOINT /home/shikina/.local/bin/jupyter notebook 
