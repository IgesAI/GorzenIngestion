# Multi-stage build for Railway deployment
FROM python:3.9-slim as python-base

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node.js stage
FROM node:18-alpine as node-base

# Install system dependencies for Python
RUN apk add --no-cache python3 python3-dev py3-pip gcc musl-dev

# Copy Python dependencies from previous stage
COPY --from=python-base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=python-base /usr/local/bin /usr/local/bin

# Set up app directory
WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Build the Next.js app
RUN npm run build

# Expose port
EXPOSE 3000

# Make sure Python is in PATH
ENV PATH="/usr/local/bin:$PATH"
ENV PYTHONPATH="/usr/local/lib/python3.9/site-packages"

# Start the application
CMD ["npm", "start"]
