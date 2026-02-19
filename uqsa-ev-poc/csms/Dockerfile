# Using the same OQS-enabled base strategy for the CSMS
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y build-essential cmake git libtool automake autoconf wget && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/liboqs.git \
    && cmake -S liboqs -B liboqs/build -DOQS_USE_OPENSSL=OFF && cmake --build liboqs/build --parallel $(nproc) && cmake --install liboqs/build
RUN git clone --depth 1 --branch openssl-3.3.0 https://github.com/openssl/openssl.git openssl-src \
    && cd openssl-src && ./config --prefix=/opt/openssl && make -j$(nproc) && make install
RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/oqs-provider.git \
    && cd oqs-provider && cmake -S . -B build -DOPENSSL_ROOT_DIR=/opt/openssl -DCMAKE_INSTALL_PREFIX=/opt/openssl && cmake --build build --parallel $(nproc) && cmake --install build

FROM python:3.11-slim-bookworm
COPY --from=builder /opt/openssl /opt/openssl
COPY --from=builder /usr/local/lib/liboqs* /usr/local/lib/
ENV PATH="/opt/openssl/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openssl/lib64:/opt/openssl/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV OPENSSL_CONF="/opt/openssl/ssl/openssl.cnf"
ENV OPENSSL_MODULES="/opt/openssl/lib64/ossl-modules"

RUN printf "openssl_conf = default_conf\n\n[default_conf]\nssl_conf = ssl_module\nproviders = provider_sect\n\n[ssl_module]\nsystem_default = system_default_sect\n\n[system_default_sect]\nCipherString = DEFAULT@SECLEVEL=0\nGroups = p256_kyber768:x25519_kyber768\n\n[provider_sect]\ndefault = default_sect\noqs = oqs_sect\n\n[default_sect]\nactivate = 1\n\n[oqs_sect]\nactivate = 1\nmodule = /opt/openssl/lib64/ossl-modules/oqsprovider.so\n" > /opt/openssl/ssl/openssl.cnf

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "csms.py"]
