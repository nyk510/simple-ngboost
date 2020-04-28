FROM python:3

ADD requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

ENV USER_NAME=ngboost
RUN groupadd -g 1000 developer &&\
  useradd -g developer -G sudo -m -s /bin/bash ${USER_NAME}

USER ${USER_NAME}
