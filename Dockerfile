# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies for PyTorch and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run directly from the src directory to fix import issues
WORKDIR /app/src

# Run the Flask app directly
CMD ["python", "app.py"]
