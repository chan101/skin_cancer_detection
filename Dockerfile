# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Run your Python script
CMD ["python", "mel.py"]
