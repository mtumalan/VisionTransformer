# Use the official Nginx image
FROM nginx:1.23-alpine

# Remove the default Nginx config (optional but recommended)
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx.conf into the container
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80 for load-balanced traffic
EXPOSE 80

# No CMD needed because the base image already runs "nginx -g 'daemon off;'"
