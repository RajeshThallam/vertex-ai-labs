

import numpy as np
import tritonclient.http as triton_http

# Set up HTTP client.
http_client = triton_http.InferenceServerClient(
    url = 'localhost:8000',
    verbose = False,
    concurrency = 5
)

features = 4
samples = 1
data = np.random.rand(samples, features).astype('float32')


triton_input_http = triton_http.InferInput(
    'mux_in',
    (samples, features),
    'FP32'
)
triton_input_http.set_data_from_numpy(data, binary_data=True)


# Set up Triton input and output objects for HTTP
outputs = []

triton_output_http_1 = triton_http.InferRequestedOutput(
    'mux_xgb_out',
     binary_data = True
)
outputs.append(triton_output_http_1)

triton_output_http_2 = triton_http.InferRequestedOutput(
    'mux_tf_out',
    binary_data = True
)
outputs.append(triton_output_http_2)

triton_output_http_3 = triton_http.InferRequestedOutput(
    'mux_sci_1_out',
     binary_data = True
)
outputs.append(triton_output_http_3)

triton_output_http_4 = triton_http.InferRequestedOutput(
    'mux_sci_2_out',
     binary_data = True
)
outputs.append(triton_output_http_4)

# Submit inference request
request_http = http_client.infer(
    'mux',
    model_version = '1',
    inputs = [triton_input_http],
    outputs = outputs
)

# Get results as numpy arrays

results = {}
for model in ["mux_xgb_out", "mux_tf_out", "mux_sci_1_out", "mux_sci_2_out"]:
   results[model] = request_http.as_numpy(model)

for model in ["mux_xgb_out", "mux_tf_out", "mux_sci_1_out", "mux_sci_2_out"]:
   print(model + " " + str(results[model]) + '\n')

