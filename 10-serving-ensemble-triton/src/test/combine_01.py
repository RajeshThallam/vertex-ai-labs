

import numpy
import tritonclient.http as triton_http

# Set up both HTTP and GRPC clients. Note that the GRPC client is generally
# somewhat faster.
http_client = triton_http.InferenceServerClient(
    url='localhost:8000',
    verbose=False,
    concurrency=5
)

# Generate example data to classify
features = 1
samples = 1
data = numpy.random.rand(features).astype('float32')
tf_data = numpy.random.rand(samples, features).astype('float32')

print("data.shape:" + str(data.shape))

# Set up Triton input and output objects for HTTP
inputs = []
triton_input_http_1 = triton_http.InferInput(
    'xgb_class',
    [features],
    'FP32'
)
triton_input_http_1.set_data_from_numpy(data, binary_data=True)
inputs.append(triton_input_http_1)

triton_input_http_2 = triton_http.InferInput(
    'tf_class',
    [samples, features],
    'FP32'
)
triton_input_http_2.set_data_from_numpy(tf_data, binary_data=True)
inputs.append(triton_input_http_2)

triton_input_http_3 = triton_http.InferInput(
    'sci_1_class',
    [features],
    'FP32'
)
triton_input_http_3.set_data_from_numpy(data, binary_data=True)
inputs.append(triton_input_http_3)

triton_input_http_4 = triton_http.InferInput(
    'sci_2_class',
    [features],
    'FP32'
)
triton_input_http_4.set_data_from_numpy(data, binary_data=True)
inputs.append(triton_input_http_4)

triton_output_http = triton_http.InferRequestedOutput(
    'OUTPUT0',
    binary_data=True
)


# Submit inference request
request_http = http_client.infer(
    'combine',
    model_version='1',
    inputs=inputs,
    outputs=[triton_output_http]
)

# Get results as numpy arrays
result_http = request_http.as_numpy('OUTPUT0')

print(result_http)
