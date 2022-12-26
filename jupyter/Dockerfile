FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

ARG UID=1000
ARG GID=1000
ARG USERNAME=shikina
ARG GROUPNAME=shikina

RUN apt update && apt upgrade -y && apt install pixz firefox-geckodriver python3-pip -y

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

COPY analyze.ipynb /opt/app/
RUN chown -R $USERNAME:$GROUPNAME /opt/app

USER $USERNAME

RUN pip install -U pip
RUN pip install numpy pandas notebook ipywidgets 'bokeh<3' bson selenium networkx scipy scikit-learn

RUN mkdir /opt/app/analysis

RUN mkdir /home/shikina/.jupyter/
COPY .jupyter/ /home/shikina/.jupyter/
ENTRYPOINT /home/shikina/.local/bin/jupyter notebook 
