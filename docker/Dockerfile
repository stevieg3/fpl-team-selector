# Start with a base image
FROM python:3.7-stretch

# Copy our application code
WORKDIR /app
COPY . .
COPY requirements.txt .

# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start the app
CMD ["gunicorn", "app.fpl_team_selector:app", "--bind", "0.0.0.0:5000"]