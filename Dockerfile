FROM python:3.9.7-slim-buster
WORKDIR /bot
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "main.py"]
