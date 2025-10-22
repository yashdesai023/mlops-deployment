FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (app code, static, templates, model)
COPY . .

# Expose port for Flask (Gunicorn will bind to this)
# 5000 is the traditional port for Flask
EXPOSE 5000

# Run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app_flask:app"]