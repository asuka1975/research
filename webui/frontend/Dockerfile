# FROM node:latest as builder

# WORKDIR /usr/src/app

# COPY ./setting/ ../app
# RUN npm install
# RUN npm run build

FROM nginx:latest as runner

WORKDIR /opt/app

# COPY --from=builder /usr/src/app/build /opt/app
COPY ./nginx.conf /etc/nginx/nginx.conf

CMD ["nginx", "-g", "daemon off;", "-c", "/etc/nginx/nginx.conf"]