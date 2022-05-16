FROM docker.io/luigi311/low-power-image-processing-base-image:latest

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install dependencies as root
RUN pip install -r requirements.txt

COPY . /app

# Fix return
RUN sed -i 's/\r$//' download_models.sh && \
    sed -i 's/\r$//' entrypoint.sh

RUN chmod +x download_models.sh && ./download_models.sh

# Entrypoint entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
