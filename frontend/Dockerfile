# 1. Use official Node.js image
FROM node:24-alpine AS builder

# 2. Set working directory
WORKDIR /app

# 3. Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# 4. Copy the rest of the frontend code
COPY . .

# 5. Build the Next.js app
RUN npm run build

# 6. Use a lightweight image for serving
FROM node:24-alpine AS runner
WORKDIR /app

# 7. Copy built files and node_modules from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

# 8. Expose port 3000
EXPOSE 3000

# 9. Start Next.js production server
CMD ["npm", "start"]