These are the well-studied ACAXU benchmarks from "Reluplex: An efficient SMT solver for verifying deep neural networks" by Katz, Guy, et al., CAV 2017.

To run a specific bencmark, such as network 3-3 with property 9, execute 'python3 -m nnenum.nnenum -o data/ACASXU_run2a_3_3_batch_2000.onnx -v data/prop_9.vnnlib'

To run all the benchmarks, run acasxu_all.py (results summary file is placed in results folder).
