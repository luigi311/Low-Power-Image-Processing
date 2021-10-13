FROM luigi311/low-power-image-processing-base-image:latest

COPY --chown=app_user:app_user requirements.txt /app/requirements.txt

WORKDIR /app

# Install dependencies as root
RUN sudo pip install -r requirements.txt

COPY --chown=app_user:app_user . /app

RUN chmod +x download_models.sh && ./download_models.sh

# Entrypoint entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]