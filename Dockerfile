# Dockerfile
#FROM python:3.11
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update
# libgl1 needed by cv2
RUN apt-get install libgl1 poppler-utils tesseract-ocr -y

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the FastAPI app
COPY ./main.py /app

# Create a directory for uploaded files
RUN mkdir /app/uploads

# Expose the port the app runs on
EXPOSE 7000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
