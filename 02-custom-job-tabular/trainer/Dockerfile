
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-4

WORKDIR /trainer
RUN pip install cloudml-hypertune

# Copies the trainer code to the docker image.
COPY train.py .

ENTRYPOINT ["python", "train.py"]
