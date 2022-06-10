FROM nvcr.io/nvidia/rapidsai/rapidsai:22.04-cuda11.2-base-ubuntu18.04

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rapids \
    && pip install -U gcsfs

RUN mkdir /train
WORKDIR /train

ADD train.py /train
ADD train.sh /train

ENTRYPOINT ["bash", "train.sh"]