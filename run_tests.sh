#!/bin/bash -e

echo "Running tests one by one. If 'Passed all tests.' is printed at then end, then we were successful."

python3 -m nnenum.nnenum -o examples/test/test_sat.onnx -v examples/test/test_prop.vnnlib -t 60 -f out.txt
grep "violated" out.txt

python3 -m nnenum.nnenum -o examples/test/test_unsat.onnx -v examples/test/test_prop.vnnlib -t 60 -f out.txt
grep "holds" out.txt

python3 -m nnenum.nnenum -o examples/acasxu/data/ACASXU_run2a_1_1_batch_2000.onnx -v examples/acasxu/data/prop_1.vnnlib

python3 -m nnenum.nnenum -o examples/mnistfc/mnist-net_256x2.onnx -v examples/mnistfc/prop_0_0.03.vnnlib

python3 -m nnenum.nnenum -o examples/mnistfc/mnist-net_256x2.onnx -v examples/mnistfc/prop_2_0.03.vnnlib

python3 -m nnenum.nnenum -o examples/cifar2020/cifar10_2_255_simplified.onnx -v examples/cifar2020/cifar10_spec_idx_11_eps_0.00784_n1.vnnlib

python3 -m nnenum.nnenum -o examples/cifar2020/cifar10_2_255_simplified.onnx -v examples/cifar2020/cifar10_spec_idx_3_eps_0.00784_n1.vnnlib -t 60 -f /dev/null

echo "Passed all tests."
