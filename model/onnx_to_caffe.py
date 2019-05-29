import onnx
import caffe2.python.onnx.backend as backend
import caffe2.python.onnx.frontend as frontend
import numpy as np
import onnx

model = onnx.load("swiftnet.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

caffe2Rep_model = backend.prepare(model, device="CUDA:0") # or "CPU"
print(type(caffe2Rep_model))
outputs = caffe2Rep_model.run(np.random.randn(1, 3, 1024, 2048).astype(np.float32))


data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 1024, 2048)
value_info = {
    'data': (data_type, data_shape)
}

# import pdb; pdb.set_trace()
print(caffe2Rep_model.predict_net.external_input)

onnx_model = frontend.caffe2_net_to_onnx_model(
    caffe2Rep_model.predict_net,
    caffe2Rep_model.init_net,
    {},
    #value_info,
)




