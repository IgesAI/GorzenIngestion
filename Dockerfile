# Railway deployment - supports both Python and Node.js
FROM python:3.11-slim

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Node.js package files and install
COPY package*.json ./
RUN npm install --production

# Copy application code
COPY . .

# Build Next.js application
RUN npm run build

# Expose port (Railway will set this automatically)
EXPOSE 3000

# Ensure Python and Node are in PATH
ENV PATH="/usr/local/bin:/usr/bin:$PATH"

# Start the application
CMD ["npm", "start"]
