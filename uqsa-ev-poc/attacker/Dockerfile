FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y hping3 python3.11 python3-pip \
    && pip3 install --break-system-packages websockets \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
CMD ["python3", "attack.py"]
