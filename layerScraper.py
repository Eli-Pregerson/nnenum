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
focus = {}

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
            key = str(ops.difference(Optimized))
            fixableDic[key] += 1
            if key == "set()":
                with open('vnncomp2025_benchmarks/all_results.csv', 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6 and str(model_path)[1:] in parts[1]:
                            if parts[4] == "sat": # or parts[4] == "unsat":
                                focus[str(model_path) + parts[2]] = float(parts[5])

# print(optDic)
# print(wDic)
# print(bDic)
# print(unknownDic)
# ./vnncomp2025_benchmarks/benchmarks/safenlp_2024/onnx/ruarobot/perturbations_0.onnx/home/elipregerson/nnenum/vnncomp2025_benchmarks/benchmarks/safenlp_2024/vnnlib/ruarobot/hyperrectangle_3689.vnnlib
# python -m nnenum.nnenum -o ../examples/convHelp/unsat/perturbations_0.onnx -v ../examples/convHelp/unsat/hyperrectangle_3689.vnnlib
# Forcing brute force goes 48 seconds to 12 mins

maxim = 0.0
for ops, num in focus.items():
    if num > maxim:
        maxim = num
        print(f"{num}: {ops}")

