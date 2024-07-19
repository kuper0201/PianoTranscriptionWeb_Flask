FROM tensorflow/tensorflow:2.15.0

COPY flask flask

RUN pip install --upgrade pip
RUN pip3 install --ignore-installed librosa pretty_midi flask pydub numpy==1.26.4

EXPOSE 5000 

WORKDIR flask
CMD ["python3", "file.py"]
ENTRYPOINT ["python3", "file.py"]