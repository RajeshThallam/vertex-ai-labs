
FROM tensorflow/serving:2.4.0

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
ENV MODEL_NAME=model

RUN mkdir -p ${MODEL_BASE_PATH}/${MODEL_NAME}/1

# copy the model file
COPY model ${MODEL_BASE_PATH}/${MODEL_NAME}/1/

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=5000 --rest_api_port=8080 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/predictor.sh \
&& chmod +x /usr/bin/predictor.sh

EXPOSE 5000
EXPOSE 8080

# Remove entrypoint from parent image
ENTRYPOINT []

CMD ["/usr/bin/predictor.sh"]
