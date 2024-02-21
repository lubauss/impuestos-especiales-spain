# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches, use environment variables
CMD ["streamlit", "run", "Home.py"]
