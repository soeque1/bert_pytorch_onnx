# References
# https://github.com/onnx/onnx/blob/master/onnx/examples/np_array_tensorproto.ipynb
# https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb

import json
import numpy as np
import requests
from google.protobuf.json_format import MessageToJson
from onnx import numpy_helper
from assets import predict_pb2

from bert_to_onnx_fixed_seq import make_inference_dummy_input


def main():
    request_headers = {'Content-Type': 'application/json'}
    inference_url = "http://0.0.0.0:8001/v1/models/bert/versions/1:predict"

    inf_dummy_input = make_inference_dummy_input()

    message_data = {"inputs": {'input_ids':inf_dummy_input[0],
                    'token_type_ids':inf_dummy_input[1],
                    'attention_mask':inf_dummy_input[2]}
    }

    input_tensors = []
    for i in inf_dummy_input:
        input_tensors.append(numpy_helper.from_array(np.array(i)))

    request_message = predict_pb2.PredictRequest()
    name_list = ['input_ids', 'token_type_ids', 'attention_mask']
    assert len(input_tensors) == len(name_list)
    for name, i in zip(name_list, input_tensors):
        # TODO: using CopyForm
        # request_message.inputs[name].CopyFrom(i)
        request_message.inputs[name].data_type = i.data_type
        request_message.inputs[name].dims.extend(i.dims)
        request_message.inputs[name].raw_data = i.raw_data

    message_data = MessageToJson(request_message)
    json_str = MessageToJson(request_message, use_integers_for_enums=True)
    message_data = json.loads(json_str)
    response = requests.post(inference_url, headers=request_headers, json=message_data)

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)
    output = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
    return output

if __name__ == '__main__':
    print(main())

