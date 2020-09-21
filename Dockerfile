FROM python:3.7-slim
RUN apt-get update && apt-get install -y libgomp1
WORKDIR /deploy/
COPY ./requirements.txt /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
COPY ./model_objects.pkl /deploy/
COPY ./score_code.py /deploy/
COPY ./app.py /deploy/
EXPOSE 8888
ENTRYPOINT ["python", "app.py"]
