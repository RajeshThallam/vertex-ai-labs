
import numpy as np
import sys
import json
import triton_python_backend_utils as pb_utils
import transformers

class TritonPythonModel:

    def initialize(self, args):
        self.out_dtypes = {}
        self.model_config = model_config = json.loads(args['model_config'])
        mux_xgb_out_config  = pb_utils.get_output_config_by_name(model_config, "mux_xgb_out")
        self.out_dtypes["mux_xgb_out"] =  pb_utils.triton_string_to_numpy(mux_xgb_out_config["data_type"])
        mux_tf_out_config  = pb_utils.get_output_config_by_name(model_config, "mux_tf_out")
        self.out_dtypes["mux_tf_out"] =  pb_utils.triton_string_to_numpy(mux_tf_out_config["data_type"])
        mux_sci_1_out_config  = pb_utils.get_output_config_by_name(model_config, "mux_sci_1_out")
        self.out_dtypes["mux_sci_1_out"] =  pb_utils.triton_string_to_numpy(mux_sci_1_out_config["data_type"])
        mux_sci_2_out_config  = pb_utils.get_output_config_by_name(model_config, "mux_sci_2_out")
        self.out_dtypes["mux_sci_2_out"] =  pb_utils.triton_string_to_numpy(mux_sci_1_out_config["data_type"])


    def execute(self, requests):
        responses = []
        for request in requests:
            mux_in = pb_utils.get_input_tensor_by_name(request, "mux_in")
            out_tensors = []
            for model in ["mux_xgb_out", "mux_tf_out", "mux_sci_1_out", "mux_sci_2_out"]:
                out_tensors.append(pb_utils.Tensor(model, mux_in.as_numpy().astype(self.out_dtypes[model])))
            inference_response = pb_utils.InferenceResponse(output_tensors = out_tensors)
            responses.append(inference_response)
        return responses



