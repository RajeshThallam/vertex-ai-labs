

import numpy
import tritonclient.http as triton_http

http_client = triton_http.InferenceServerClient(
    url='localhost:8000',
    verbose=False,
    concurrency=5
)

features = 4
samples = 1
data = numpy.random.rand(samples, features).astype('float32')

triton_input_http = triton_http.InferInput(
    'INPUT0',
    (samples, features),
    'FP32'
)
triton_input_http.set_data_from_numpy(data, binary_data=True)

triton_output_http = triton_http.InferRequestedOutput(
    'OUTPUT0',
    binary_data=True
)


request_http = http_client.infer(
    'ensemble',
    model_version='1',
    inputs=[triton_input_http],
    outputs=[triton_output_http]
)

result_http = request_http.as_numpy('OUTPUT0')

print(result_http)
