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
python bert_to_onnx.py
```

### Tests
```
python -m pytest tests
```