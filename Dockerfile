FROM python:3.9.16

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Install mukkeBude
RUN pip install .

# Install Flask-Webapp
WORKDIR /app/flask-webapp
RUN pip install -r requirements-dev.txt

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
