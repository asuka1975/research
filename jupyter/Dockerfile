FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN apt update && apt upgrade -y && apt install pixz firefox-geckodriver python3-pip -y

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

COPY analyze.ipynb /opt/app/
RUN chown -R $USERNAME:$GROUPNAME /opt/app

USER $USERNAME

RUN pip install -U pip
RUN pip install numpy pandas notebook ipywidgets bokeh bson selenium networkx scipy scikit-learn

RUN mkdir /opt/app/analysis

RUN mkdir /home/$USERNAME/.jupyter/
COPY .jupyter/ /home/$USERNAME/.jupyter/
ENV UNAME=$USERNAME
ENTRYPOINT /home/$UNAME/.local/bin/jupyter notebook 
