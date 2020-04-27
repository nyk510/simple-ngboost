FROM python:3

ADD . .
RUN pip install -U pip && pip install -r requirements.txt
