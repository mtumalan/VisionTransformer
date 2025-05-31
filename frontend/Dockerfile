# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files first to leverage Docker cache
COPY package*.json ./

RUN npm install

# Copy the rest of the app
COPY . .

# Build for production
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
