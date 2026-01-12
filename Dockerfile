FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Api.py .

EXPOSE 5000

CMD ["python", "Api.py"]