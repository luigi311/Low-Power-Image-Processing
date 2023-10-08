FROM docker.io/luigi311/low-power-image-processing-base-image:latest

WORKDIR /app

COPY requirements.txt .

# Install dependencies as root
RUN pip install -r requirements.txt

COPY . .

# Fix return
RUN sed -i 's/\r$//' download_models.sh && \
    sed -i 's/\r$//' entrypoint.sh

RUN chmod +x download_models.sh && ./download_models.sh

# Entrypoint entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
