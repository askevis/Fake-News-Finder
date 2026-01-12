FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shiny app code
COPY app.py .

EXPOSE 8000

# Command to run Shiny for Python
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]