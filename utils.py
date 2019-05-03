import onnx

def update_inputs_outputs_dims(model, input_dims, output_dims):
    """
        This function updates the sizes of dimensions of the model's inputs and outputs to the values
        provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
        will be set for that dimension.
    """
    def update_dim(tensor, dim, i, j, dim_param_prefix):
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                dim_proto.dim_value = dim
            else:
                dim_proto.dim_param = dim_param_prefix + str(i) + '_' + str(j)
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

    for i, input_dim_arr in enumerate(input_dims):
        for j, dim in enumerate(input_dim_arr):
            update_dim(model.graph.input[i], dim, i, j, 'in_')

    for i, output_dim_arr in enumerate(output_dims):
        for j, dim in enumerate(output_dim_arr):
            update_dim(model.graph.output[i], dim, i, j, 'out_')

    onnx.checker.check_model(model)
    return model

