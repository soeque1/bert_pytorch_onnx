import numpy as np
import torch
import onnxruntime

from pytorch_pretrained_bert import BertModel

def make_train_dummy_input():
    org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
    org_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
    org_input_mask = torch.LongTensor([[0, 0, 1, 1]])
    return (org_input_ids, org_token_type_ids, org_input_mask)

def make_inference_dummy_input():
    inf_input_ids = [[31, 51, 98, 1]]
    inf_token_type_ids = [[1, 1, 1, 1]]
    inf_input_mask = [[0, 0, 1, 1]]
    return (inf_input_ids, inf_token_type_ids, inf_input_mask)


if __name__ == '__main__':
    MODEL_ONNX_PATH = "./onnx/torch_bert_fixed.onnx"
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX

    model = BertModel.from_pretrained('bert-base-uncased')
    model.train(False)

    org_dummy_input = make_train_dummy_input()
    inf_dummy_input = make_inference_dummy_input()

    output = torch.onnx.export(model,
                               org_dummy_input,
                               MODEL_ONNX_PATH,
                               verbose=True,
                               operator_export_type=OPERATOR_EXPORT_TYPE,
                               input_names=['input_ids', 'token_type_ids', 'attention_mask'],
                               output_names=['output']
                               )
    print("Export of torch_model.onnx complete!")

    print(model(*(torch.LongTensor(i) for i in inf_dummy_input))[0][0][0:5])

    sess = onnxruntime.InferenceSession(MODEL_ONNX_PATH)
    pred_onnx = sess.run(None, {'input_ids':np.array(inf_dummy_input[0]),
                                'token_type_ids':np.array(inf_dummy_input[1]),
                                'attention_mask':np.array(inf_dummy_input[2])})
    print(pred_onnx[0][0:5])