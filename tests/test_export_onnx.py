import unittest
import pytest
import torch
from torch.autograd import Variable
import onnx
import numpy as np
import onnxruntime
from pytorch_pretrained_bert import BertConfig, BertModel
from models.base import Model
from models.bert_custom import BertSelfAttention_custom, BertEmbeddings_custom
from models.bert_custom import BertModel_emb_custom, BertModel_emb_encoder_custom, BertModel_custom

ONNX_FOLDER = "./onnx/"
OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
BERT_CONFIG_PATH = "./models/bert_config.json"


class TestBase(unittest.TestCase):
    def setUp(self):
        org_input_shape = (3, 100, 100)
        inf_input_shape = (3, 25, 25)
        self.model_onnx_path = ONNX_FOLDER + "torch_model.onnx"
        self.org_dummy_input = Variable(torch.randn(1, *org_input_shape))
        self.inf_dummy_input = Variable(torch.randn(1, *inf_input_shape))


    def test_convert_onnx(self):
        model = Model()
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'0':np.array(self.inf_dummy_input)})
        print(pred_onnx[0][0:5])


class TestBertModuleAttention(unittest.TestCase):
    def setUp(self):
        org_seq_len = 4
        hidden_size = 32
        org_hidden_shape = (org_seq_len, hidden_size)
        org_mask_shape = (1, 1, org_seq_len)

        inf_seq_len = 2
        inf_hidden_shape = (inf_seq_len, hidden_size)
        inf_mask_shape = (1, 1, inf_seq_len)

        self.model_onnx_path = ONNX_FOLDER + "torch_self_aten_model.onnx"
        self.org_dummy_input = (Variable(torch.randn(1, *org_hidden_shape)),
                                Variable(torch.randn(1, *org_mask_shape)))
        self.inf_dummy_input = (Variable(torch.randn(1, *inf_hidden_shape)),
                                Variable(torch.randn(1, *inf_mask_shape)))

    def test_convert_onnx(self):
        model = BertSelfAttention_custom(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['hidden_states', 'attention_mask']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'hidden_states':np.array(self.inf_dummy_input[0]),
                                    'attention_mask':np.array(self.inf_dummy_input[1])})
        print(pred_onnx[0][0:5])


class TestBertModuleEmbedding(unittest.TestCase):
    def setUp(self):
        org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
        org_token_type_ids = torch.LongTensor([0, 0, 1, 0])
        org_position_ids = torch.LongTensor(torch.ones_like(org_input_ids[0]))
        org_position_ids = org_position_ids.cumsum(dim=0) - 1
        self.org_dummy_input = (org_input_ids, org_token_type_ids, org_position_ids)

        inf_input_ids = torch.LongTensor([[31, 51, 98, 1]])
        inf_token_type_ids = torch.LongTensor([0, 0, 1, 0])
        inf_position_ids = torch.ones_like(inf_input_ids[0])
        inf_position_ids = inf_position_ids.cumsum(dim=0) - 1
        self.inf_dummy_input = (inf_input_ids, inf_token_type_ids, inf_position_ids)

        self.model_onnx_path = ONNX_FOLDER + "torch_bert_emb_model.onnx"

    def test_convert_onnx(self):
        model = BertEmbeddings_custom(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['input_ids', 'token_type_ids', 'position_ids']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'input_ids':np.array(self.inf_dummy_input[0]),
                                    'token_type_ids':np.array(self.inf_dummy_input[1]),
                                    'position_ids':np.array(self.inf_dummy_input[2])})
        print(pred_onnx[0][0:5])


class TestBertModelOrg(unittest.TestCase):
    def setUp(self):
        org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
        org_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
        org_input_mask = torch.LongTensor([[0, 0, 1, 1]])
        self.org_dummy_input = (org_input_ids, org_token_type_ids, org_input_mask)

        inf_input_ids = torch.LongTensor([[31, 51, 98, 1]])
        inf_token_type_ids = torch.LongTensor([[0, 0, 1, 1]])
        inf_input_mask = torch.LongTensor([[0, 0, 1, 1]])
        self.inf_dummy_input = (inf_input_ids, inf_token_type_ids, inf_input_mask)
        self.model_onnx_path = ONNX_FOLDER + "torch_integ_bert_org_model.onnx"

    def test_convert_onnx(self):
        model = BertModel(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['input_ids', 'token_type_ids', 'attention_mask']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_embedding_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'input_ids':np.array(self.inf_dummy_input[0]),
                                    'token_type_ids':np.array(self.inf_dummy_input[1]),
                                    'attention_mask':np.array(self.inf_dummy_input[2])})
        print(pred_onnx[0][0:5])


class TestBertModelEmbedding(unittest.TestCase):
    def setUp(self):
        org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
        org_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
        org_input_mask = torch.LongTensor([[0, 0, 1, 1]])
        org_position_ids = torch.LongTensor(torch.ones_like(org_input_ids[0]))
        org_position_ids = org_position_ids.cumsum(dim=0) - 1
        self.org_dummy_input = (org_input_ids, org_token_type_ids, org_input_mask, org_position_ids)

        inf_input_ids = torch.LongTensor([[31, 51, 98]])
        inf_token_type_ids = torch.LongTensor([[0, 0, 1]])
        inf_input_mask = torch.LongTensor([[0, 0, 1]])
        inf_position_ids = torch.ones_like(inf_input_ids[0])
        inf_position_ids = inf_position_ids.cumsum(dim=0) - 1
        self.inf_dummy_input = (inf_input_ids, inf_token_type_ids, inf_input_mask, inf_position_ids)

        self.model_onnx_path = ONNX_FOLDER + "torch_integ_bert_emb_model.onnx"


    def test_convert_embedding_onnx(self):
        model = BertModel_emb_custom(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['input_ids', 'token_type_ids', 'attention_mask', 'position_ids']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_embedding_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'input_ids':np.array(self.inf_dummy_input[0]),
                                    'token_type_ids':np.array(self.inf_dummy_input[1]),
                                    'attention_mask':np.array(self.inf_dummy_input[2]),
                                    'position_ids':np.array(self.inf_dummy_input[3])})
        print(pred_onnx[0][0:5])


    def test_convert_embedding_and_encoder_onnx(self):
        model = BertModel_emb_encoder_custom(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['input_ids', 'token_type_ids', 'attention_mask', 'position_ids']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_embedding_and_encoder_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'input_ids':np.array(self.inf_dummy_input[0]),
                                    'token_type_ids':np.array(self.inf_dummy_input[1]),
                                    'attention_mask':np.array(self.inf_dummy_input[2]),
                                    'position_ids':np.array(self.inf_dummy_input[3])})
        print(pred_onnx[0][0:5])


    def test_convert_embedding_and_encoder_pooling_onnx(self):
        model = BertModel_custom(BertConfig.from_json_file(BERT_CONFIG_PATH))
        model.train(False)

        output = torch.onnx.export(model,
                                   self.org_dummy_input,
                                   self.model_onnx_path,
                                   verbose=True,
                                   operator_export_type=OPERATOR_EXPORT_TYPE,
                                   input_names=['input_ids', 'token_type_ids', 'attention_mask', 'position_ids']
                                   )
        print("Export of torch_model.onnx complete!")


    def test_inference_embedding_and_encoder_pooling_onnx(self):
        onnx_model = onnx.load(self.model_onnx_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        pred_onnx = sess.run(None, {'input_ids':np.array(self.inf_dummy_input[0]),
                                    'token_type_ids':np.array(self.inf_dummy_input[1]),
                                    'attention_mask':np.array(self.inf_dummy_input[2]),
                                    'position_ids':np.array(self.inf_dummy_input[3])})
        print(pred_onnx[0][0:5])



if __name__ == '__main__':
    unittest.main()