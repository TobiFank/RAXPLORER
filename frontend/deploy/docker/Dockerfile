# frontend/deploy/docker/Dockerfile
FROM node:20-slim

WORKDIR /app

# Install dependencies only when needed
COPY package.json yarn.lock ./
RUN yarn install

# Copy the rest of the application
COPY . .

# Next.js collects completely anonymous telemetry data about general usage.
# Learn more here: https://nextjs.org/telemetry
# Uncomment the following line in case you want to disable telemetry during the build.
# ENV NEXT_TELEMETRY_DISABLED 1

# Expose Next.js default port
EXPOSE 3000

# Start development server
CMD ["yarn", "dev"]