

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
features = 4
samples = 1
data = numpy.random.rand(samples, features).astype('float32')

# Set up Triton input and output objects for HTTP
triton_input_http = triton_http.InferInput(
    'input__0',
    (samples, features),
    'FP32'
)
triton_input_http.set_data_from_numpy(data, binary_data=True)

triton_output_http = triton_http.InferRequestedOutput(
    'output__0',
    binary_data=True
)


# Submit inference request
request_http = http_client.infer(
    'sci_2',
    model_version='1',
    inputs=[triton_input_http],
    outputs=[triton_output_http]
)

# Get results as numpy arrays
result_http = request_http.as_numpy('output__0')

print(result_http)
