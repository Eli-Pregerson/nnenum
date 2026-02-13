#!/usr/bin/env python3
"""
Script to add a Reshape layer at the end of an ONNX model
"""

import onnx
from onnx import helper, numpy_helper
import numpy as np

# Load the model
model_path = "examples/convHelp/perturbations_0.onnx"
final_path = "examples/convHelp/perturbations_0_reshape.onnx"
model = onnx.load(model_path)

# Get the current output
current_output = model.graph.output[0]
print(f"Current output: {current_output.name}")
print(f"Current output shape: {current_output.type.tensor_type.shape}")

# Create a new intermediate output name (rename current output)
intermediate_output_name = current_output.name + "_pre_reshape"

# Get output shape dimensions
output_shape = []
for dim in current_output.type.tensor_type.shape.dim:
    if dim.dim_value:
        output_shape.append(dim.dim_value)
    else:
        output_shape.append(-1)  # dynamic dimension

print(f"Parsed output shape: {output_shape}")

# Calculate total elements for flattening (example: reshape to [1, -1])
# You can modify target_shape to whatever you need
target_shape = [1, -1]  # Flatten to batch_size=1, features=-1
print(f"Target reshape: {target_shape}")

# Create a Constant node to output the shape tensor
shape_const_name = current_output.name + "_reshape_shape"
shape_const_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=[shape_const_name],
    name=shape_const_name + "_node",
    value=numpy_helper.from_array(
        np.array(target_shape, dtype=np.int64),
        name=shape_const_name
    )
)
model.graph.node.append(shape_const_node)

# Update the existing output nodes to use intermediate name
for node in model.graph.node:
    for i, output in enumerate(node.output):
        if output == current_output.name:
            node.output[i] = intermediate_output_name

# Create Reshape node
reshape_node = helper.make_node(
    'Reshape',
    inputs=[intermediate_output_name, shape_const_name],
    outputs=[current_output.name],
    name=current_output.name + "_reshape_node"
)

# Add the reshape node to the graph
model.graph.node.append(reshape_node)

# No need to change graph.output - it already points to current_output.name
# which is now the output of our reshape node

# Save the modified model
onnx.save(model, final_path)
print(f"\nModel saved with Reshape layer added to: {final_path}")
print(f"New final operation: {reshape_node.name}")
