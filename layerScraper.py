import os
import onnx
from collections import defaultdict

optDic = defaultdict(int)
Optimized = {'Add', 'Sub', 'Flatten', 'MatMul', 'Relu', 'Gemm', 'Conv', 'Reshape', 'Constant'}
wDic = defaultdict(int)
Whitelist = {'Add', 'AveragePool', 'Constant', 'Concat', 'Conv', 'Flatten', 'Gather', \
                              'Gemm', 'MatMul', 'Mul', 'Reshape', 'Relu', 'Shape', 'Sub', 'Unsqueeze', 'Slice', \
                              'Dropout', 'BatchNormalization', 'ConvTranspose', 'Upsample'}
bDic = defaultdict(int)
Blacklist = {'Atan', 'MaxPool', 'Sigmoid', 'Tanh'}
unknownDic = defaultdict(int)
fixableDic = defaultdict(int)
focus = {}

# First pass: count instances per ONNX file from instances.csv files
onnx_instance_count = defaultdict(int)
benchmarks_dir = './vnncomp2025_benchmarks/benchmarks'
for benchmark_name in os.listdir(benchmarks_dir):
    benchmark_dir = os.path.join(benchmarks_dir, benchmark_name)
    instances_csv = os.path.join(benchmark_dir, 'instances.csv')
    if os.path.isfile(instances_csv):
        with open(instances_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 1:
                    onnx_rel_path = parts[0]
                    # Build full path to match against model_path
                    onnx_full_path = os.path.join(benchmark_dir, onnx_rel_path)
                    onnx_instance_count[onnx_full_path] += 1

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
            # Weight by number of instances using this ONNX file
            instance_weight = onnx_instance_count.get(model_path, 1)
            fixableDic[key] += instance_weight
            if ops.difference(Optimized) == {'Constant', 'Conv', 'Reshape'}:
                with open('vnncomp2025_benchmarks/all_results.csv', 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6 and str(model_path)[1:] in parts[1]:

                            if parts[4] == "sat" or parts[4] == "unsat":
                                focus[str(model_path) + parts[2]] = float(parts[5])

# print(optDic)
# print(wDic)
# print(bDic)
# print(unknownDic)
# ./vnncomp2025_benchmarks/benchmarks/safenlp_2024/onnx/ruarobot/perturbations_0.onnx/home/elipregerson/nnenum/vnncomp2025_benchmarks/benchmarks/safenlp_2024/vnnlib/ruarobot/hyperrectangle_3689.vnnlib
# python -m nnenum.nnenum -o ../examples/convHelp/unsat/perturbations_0.onnx -v ../examples/convHelp/unsat/hyperrectangle_3689.vnnlib
# Forcing brute force goes 48 seconds to 12 mins

maxim = 0.0
for ops, num in fixableDic.items():
    print(f"{num}: {ops}")

