FROM python:3.9-slim

WORKDIR /app

COPY app/ /app/
COPY requirements.txt /app/
COPY models/ /app/models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

CMD ["python", "app.py"]
