import os
import onnx
from collections import defaultdict

optDic = defaultdict(int)
Optimized = {'Add', 'Sub', 'Flatten', 'MatMul', 'Relu', 'Gemm'}
wDic = defaultdict(int)
Whitelist = {'Add', 'AveragePool', 'Constant', 'Concat', 'Conv', 'Flatten', 'Gather', \
                              'Gemm', 'MatMul', 'Mul', 'Reshape', 'Relu', 'Shape', 'Sub', 'Unsqueeze', 'Slice', \
                              'Dropout', 'BatchNormalization', 'ConvTranspose', 'Upsample'}
bDic = defaultdict(int)
Blacklist = {'Atan', 'MaxPool', 'Sigmoid', 'Tanh'}
unknownDic = defaultdict(int)
fixableDic = defaultdict(int)


for dirpath, dirnames, filenames in os.walk('./vnncomp2025_benchmarks/benchmarks'):
    for file in filenames:
        if file.endswith('.onnx'):
            model_path = os.path.join(dirpath, file)
            onnx_model = onnx.load(model_path)
            ops = set()
            for node in onnx_model.graph.node:
                ops.add(node.op_type)
            for op in ops:
                if op in Optimized:
                    optDic[op] += 1
                elif op in Whitelist:
                    wDic[op] += 1
                elif op in Blacklist:
                    bDic[op] += 1
                else:
                    unknownDic[op] += 1
            fixableDic[str(ops.difference(Optimized))] += 1

# print(optDic)
# print(wDic)
# print(bDic)
# print(unknownDic)
for ops, num in fixableDic.items():
    print(f"{num}: {ops}")

