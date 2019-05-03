import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.onnx as torch_onnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        print(x.shape)
        x = x.view(x.size()[0], x.size()[1], -1)
        x = x.reshape(x.size()[1], x.size()[0], -1)
        print(x.shape)
        return torch.mean(x, dim=2)


if __name__ == '__main__':
    # Use this an input trace to serialize the model
    input_shape = (3, 100, 100)
    model_onnx_path = "torch_model.onnx"
    model = Model()
    model.train(False)

    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    output = torch_onnx.export(model,
                               dummy_input,
                               model_onnx_path,
                               verbose=True)
    print("Export of torch_model.onnx complete!")