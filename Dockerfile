FROM python:3.10-slim-buster

WORKDIR /app

COPY ./requirements.txt .
COPY ./cat_vs_dog_classifier.py .
COPY ./cats-and-dogs.keras .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "cat_vs_dog_classifier.py", "--server.port=8501", "--server.address=0.0.0.0"]