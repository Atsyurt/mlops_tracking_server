FROM python:3.12.10

WORKDIR /app

# Install mlflow and packages requied to interact with PostgreSQL and MinIO
RUN pip install psycopg2
RUN pip install mlflow
RUN pip install boto3

