FROM python:3.8-buster as builder

WORKDIR /opt/app

COPY requirements.lock /opt/app
RUN pip3 install -r requirements.lock

FROM python:3.8-slim-buster as runner

COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin/uwsgi /usr/local/bin/uwsgi

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

RUN apt update \
    && apt install -y libxml2 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/app/
RUN mkdir -p /opt/app/logs
RUN touch /opt/app/logs/server.log
COPY main.py /opt/app
COPY uwsgi.ini /opt/app
RUN chown -R 1000 /opt/app

EXPOSE 51223

USER $USERNAME

WORKDIR /opt/app

CMD ["uwsgi", "--ini", "uwsgi.ini"]