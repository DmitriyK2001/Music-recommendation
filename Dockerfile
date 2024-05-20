FROM ubuntu:22.04
FROM python:3.12-slim

#Python parameters
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
#pip parameters
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
#poetry parameters
ENV POETRY_VERSION=1.7.1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_CACHE_DIR='/var/cache/pypoetry'
ENV POETRY_HOME='/usr/local'

ENV DEBIAN_FRONTEND = noninteractive

RUN apt-get update -y
#installing Python
RUN apt-get install python3 -y
RUN apt-get install curl -y
#installing this for dvc
RUN apt-get install gcc python3-dev -y
#installing git
RUN apt-get update && apt-get install git -y
# Download poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

#copy necessary files from repository
COPY pyproject.toml /Music-recommendation/
COPY poetry.lock /Music-recommendation/
COPY command.py /Music-recommendation/
COPY music/ /Music-recommendation/music/
COPY .dvc/  /Music-recommendation/.dvc/
COPY .dvcignore /Music-recommendation/
COPY config/ /Music-recommendation/config/

SHELL ["/bin/bash", "--login", "-c"]
#setting up work directory
WORKDIR /Music-recommendation/
# Installing poetry
RUN poetry install
#git for dvc
RUN git init
#load data from gdrive
#it is commented, because it requires communication in browser, in can't be achieved in docker container
#RUN dvc pull
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /Music-recommendation/
USER appuser

#starting command.py file with parameter 'infer'
CMD ["python", "command.py", "infer"]
#for setting
#ENTRYPOINT ["/bin/bash"]
