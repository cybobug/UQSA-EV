# STAGE 1: Build OQS-Enabled OpenSSL & Provider
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake git libtool automake autoconf wget \
    python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# 1. Build liboqs (Master branch)
RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/liboqs.git \
    && cmake -S liboqs -B liboqs/build \
    -DOQS_USE_OPENSSL=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build liboqs/build --parallel $(nproc) \
    && cmake --install liboqs/build

# 2. Build OpenSSL 3.x
# Note: We install to /opt/openssl to keep it separate from system SSL
RUN git clone --depth 1 --branch openssl-3.3.0 https://github.com/openssl/openssl.git openssl-src \
    && cd openssl-src \
    && ./config --prefix=/opt/openssl --openssldir=/opt/openssl/ssl \
    && make -j$(nproc) \
    && make install

# 3. Build oqs-provider
# Crucial: Must point OPENSSL_ROOT_DIR to our custom build /opt/openssl
RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/oqs-provider.git \
    && cmake -S oqs-provider -B oqs-provider/build \
    -DOPENSSL_ROOT_DIR=/opt/openssl \
    -DCMAKE_INSTALL_PREFIX=/opt/openssl/lib64/ossl-modules \
    -DOQS_ROOT_DIR=/usr/local \
    && cmake --build oqs-provider/build --parallel $(nproc) \
    && cp oqs-provider/build/lib/oqsprovider.so /opt/openssl/lib64/ossl-modules/

# STAGE 2: Final Runtime Image
FROM python:3.11-slim-bookworm

# Copy Artifacts
COPY --from=builder /opt/openssl /opt/openssl
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include/oqs /usr/local/include/oqs

# Set Runtime Environment
ENV PATH="/opt/openssl/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openssl/lib64:/opt/openssl/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV OPENSSL_CONF="/opt/openssl/ssl/openssl.cnf"
ENV OPENSSL_MODULES="/opt/openssl/lib64/ossl-modules"

# Generate OpenSSL Config with PQC enabled
RUN printf "openssl_conf = default_conf\n\n\
    [default_conf]\n\
    ssl_conf = ssl_module\n\
    providers = provider_sect\n\n\
    [ssl_module]\n\
    system_default = system_default_sect\n\n\
    [system_default_sect]\n\
    CipherString = DEFAULT@SECLEVEL=0\n\
    Groups = p256_kyber768:x25519_kyber768\n\n\
    [provider_sect]\n\
    default = default_sect\n\
    oqs = oqs_sect\n\n\
    [default_sect]\n\
    activate = 1\n\n\
    [oqs_sect]\n\
    activate = 1\n\
    module = /opt/openssl/lib64/ossl-modules/oqsprovider.so\n" > /opt/openssl/ssl/openssl.cnf

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix log permissions and remove old logs
RUN mkdir -p /app/logs /app/certs && rm -f /app/logs/proxy.log && chmod 777 /app/logs

COPY . .

CMD ["python3", "proxy.py"]