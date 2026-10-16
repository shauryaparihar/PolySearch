# PolySearch web UI: build with Node, serve the static files with unprivileged nginx.
#   docker build -t polysearch-web --build-arg VITE_API_URL=http://localhost:8000 frontend
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The API address is baked into the static bundle at build time (it is not a secret).
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD wget -q --spider http://127.0.0.1:8080/ || exit 1
