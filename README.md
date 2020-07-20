# bert_pytorch_onnx

### Prepare
```
(MAC) brew install libomp
pip install -r requirements.txt
```

### config.json
According to the test of pytorch github, [the test config](https://github.com/huggingface/pytorch-pretrained-BERT/blob/68a889ee43916380f26a3c995e1638af41d75066/tests/modeling_test.py), BertModelTester's initializer is used.

### Main
```
mkdir onnx
python bert_to_onnx_fixed_seq.py
python bert_to_onnx_dynamic_seq.py
```

### (Prepare) Server
```
git clone https://github.com/microsoft/onnxruntime.git --recursive
docker build -t mcr.microsoft.com/azureml/onnxruntime:latest -f onnxruntime/dockerfiles/Dockerfile.server onnxruntime/dockerfiles/
```

#### (RUN) Server (HTTP)
```
docker run --name onnx_server -p 8003:8001 -p 8004:50051 -v $PWD/:/usr/server mcr.microsoft.com/azureml/onnxruntime:latest --log_level verbose --model_path=/usr/server/onnx/torch_bert_fixed.onnx --model_name=bert --model_version=1 --http_port 8001 --grpc_port 50051
```

#### (Test) Client
```
PYTHONPATH=./tutorials/tutorials python test_client.py
```

or

```
curl -X POST -d "@xxx.json" -H "Content-Type: application/json" http://0.0.0.0:8001/v1/models/bert/versions/1:predict
```

### Tests
```
python -m pytest tests
```

### TODO:
- gRPC