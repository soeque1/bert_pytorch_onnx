# References
# https://github.com/onnx/onnx/blob/master/onnx/examples/np_array_tensorproto.ipynb
# https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb

import grpc
import json
import numpy as np
import requests
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from onnx import numpy_helper
from assets import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from bert_to_onnx_fixed_seq import make_inference_dummy_input


def main_http():
    request_headers = {'Content-Type': 'application/json'}
    inference_url = "http://127.0.0.1:8003/v1/models/bert/versions/1:predict"

    inf_dummy_input = make_inference_dummy_input()

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

    json_str = MessageToJson(request_message)
    message_data = json.loads(json_str)
    response = requests.post(inference_url, headers=request_headers, json=message_data)

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)
    output = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
    return output

def main_grpc():
    inf_dummy_input = make_inference_dummy_input()

    input_tensors = []
    for i in inf_dummy_input:
        input_tensors.append(numpy_helper.from_array(np.array(i)))

    # from tensorflow_serving.apis import predict_pb2
    request_message = predict_pb2.PredictRequest()
    # request_message.model_spec.name = "my_model"
    # request_message.model_spec.signature_name = "serving_default"

    name_list = ['input_ids', 'token_type_ids', 'attention_mask']
    assert len(input_tensors) == len(name_list)
    for name, i in zip(name_list, input_tensors):
        # request_message.inputs[name].CopyFrom(tf.make_tensor_proto(i))
        request_message.inputs[name].data_type = i.data_type
        request_message.inputs[name].dims.extend(i.dims)
        request_message.inputs[name].raw_data = i.raw_data

    uri = "127.0.0.1:8004"

    with grpc.insecure_channel(uri) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        response = stub.Predict(request_message, 1)

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)
    output = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
    return output

def main():
    print(main_http())
    print(main_grpc())


if __name__ == '__main__':
    main()

