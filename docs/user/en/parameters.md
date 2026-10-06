# Package parameters

The names below are the keys shown under `Inspector > Parameters`. All listed core packages are version `0.1.0` and appear with their full identity, `core.<id>@0.1.0`. `No declared default` means the package schema does not specify an initial value; the Inspector and validator show whether the field is available and valid. Limits are those declared by the schema. `top` and `bottom` mean that the parameter is also shown in the corresponding row on the node card.

## Core parameters

Each parameter has its own row so the purpose of every field is clear, including fields that share a default or constraint.

| Package | Key | Type and default | Constraints, choices, position | Meaning |
| --- | --- | --- | --- | --- |
| `core.adaptive-avg-pool2d@0.1.0` | `output_size` | integer, not declared | minimum 1; `bottom` | Target spatial size used by adaptive average pooling. |
| `core.add@0.1.0` | — | — | — | No parameters. Adds two or more tensors of the same shape. |
| `core.batch-norm2d@0.1.0` | `num_features` | integer, not declared | minimum 1; `top` | Number of channels in the C dimension of the rank-4 NCHW input. |
| `core.batch-norm2d@0.1.0` | `eps` | number, `0.00001` | minimum 0 | Stabilizing constant added to variance during normalization. |
| `core.batch-norm2d@0.1.0` | `momentum` | number, `0.1` | from 0 to 1 | Weight used to update the running mean and variance. |
| `core.batch-norm2d@0.1.0` | `affine` | Boolean, `true` | — | Enables learnable scale and offset after normalization. |
| `core.batch-norm2d@0.1.0` | `track_running_stats` | Boolean, `true` | — | Keeps running mean and variance statistics for evaluation mode. |
| `core.cast@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `int32`, `int64`, `bool` | Dtype of the converted tensor; its shape is unchanged. |
| `core.concat@0.1.0` | `dim` | integer, not declared | — | Axis along which to concatenate inputs. |
| `core.conv2d@0.1.0` | `in_channels` | integer, not declared | minimum 1; `top` | Number of expected input channels. |
| `core.conv2d@0.1.0` | `out_channels` | integer, not declared | minimum 1; `bottom` | Number of channels produced by the convolution. |
| `core.conv2d@0.1.0` | `kernel_size` | integer, not declared | minimum 1 | Side length of the square convolution window. |
| `core.conv2d@0.1.0` | `stride` | integer, `1` | minimum 1 | Step between adjacent window positions. |
| `core.conv2d@0.1.0` | `padding` | integer, `0` | minimum 0 | Spatial padding added around the input. |
| `core.conv2d@0.1.0` | `dilation` | integer, `1` | minimum 1 | Spacing between elements in the window. |
| `core.conv2d@0.1.0` | `groups` | integer, `1` | minimum 1 | Number of independent channel groups; must be compatible with both channel counts. |
| `core.conv2d@0.1.0` | `bias` | Boolean, `true` | — | Includes a learnable bias term. |
| `core.cross-entropy@0.1.0` | — | — | — | No parameters. Classification loss; reads the batch target named `target` and produces a scalar loss result. |
| `core.embedding@0.1.0` | `num_embeddings` | integer, not declared | minimum 1; `top` | Number of distinct indices represented by the embedding table. |
| `core.embedding@0.1.0` | `embedding_dim` | integer, not declared | minimum 1; `bottom` | Width of the vector associated with each index. |
| `core.embedding@0.1.0` | `input_dtype` | dtype, `int64` | `int32`, `int64` | Integer dtype of indices passed to the table. |
| `core.embedding@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64` | Dtype of embedding vectors. |
| `core.flatten@0.1.0` | `start_dim` | integer, `1` | — | First dimension in the range to flatten. |
| `core.flatten@0.1.0` | `end_dim` | integer, `-1` | — | Last dimension in the range to flatten; `-1` means the final dimension. |
| `core.fork@0.1.0` | — | — | — | No parameters. Branch point that preserves the input tensor. |
| `core.horizontal-repeat@0.1.0` | `times` | integer, not declared | minimum 2 | Number of independent copies of the subflow run in parallel. |
| `core.horizontal-repeat@0.1.0` | `join` | `stereotype` reference; `core.concat@^0.1.0` with `dim: -1` | selector filtered to `join` packages | Package that combines the outputs of parallel copies; choosing another join package loads its parameters. |
| `core.input@0.1.0` | `binding` | string, not declared | — | Name of the active dataset input slot used by this root input. In subflows, the initial input is inherited from the container. |
| `core.layer-norm@0.1.0` | `normalized_shape` | integer, not declared | minimum 1; `top` | Final dimension normalized for each sample. |
| `core.layer-norm@0.1.0` | `eps` | number, `0.00001` | minimum 0 | Stabilizing constant used to calculate variance. |
| `core.layer-norm@0.1.0` | `elementwise_affine` | Boolean, `true` | — | Enables learnable element-wise scale and offset. |
| `core.linear@0.1.0` | `in_features` | integer, not declared | minimum 1; `top` | Expected width of the final input dimension. |
| `core.linear@0.1.0` | `out_features` | integer, not declared | minimum 1; `bottom` | Width of the final output dimension. |
| `core.linear@0.1.0` | `bias` | Boolean, `true` | — | Includes a learnable bias term. |
| `core.linear@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64` | Dtype of linear weights and calculations. |
| `core.loss-output@0.1.0` | — | — | — | No parameters. Terminal that collects one loss result; it does not automatically sum multiple contributions. |
| `core.matmul@0.1.0` | — | — | — | No parameters. Receives two or more tensors and applies matrix multiplication from left to right. |
| `core.max-pool2d@0.1.0` | `kernel_size` | integer, not declared | minimum 1 | Side length of the pooling window. |
| `core.max-pool2d@0.1.0` | `stride` | integer, `1` | minimum 1 | Step between successive window positions. |
| `core.max-pool2d@0.1.0` | `padding` | integer, `0` | minimum 0 | Padding added around the input before pooling. |
| `core.max-pool2d@0.1.0` | `dilation` | integer, `1` | minimum 1 | Spacing between elements of the pooling window. |
| `core.max-pool2d@0.1.0` | `ceil_mode` | Boolean, `false` | — | When enabled, rounds window calculations up; otherwise rounds down. |
| `core.mse-loss@0.1.0` | — | — | — | No parameters. MSE regression loss; uses only the package-declared target named `target`, with the specific `flatten_batch` target transform. It does not select or flatten arbitrary targets. |
| `core.output@0.1.0` | — | — | — | No parameters. Terminal marking the model prediction; accepts one edge. |
| `core.positional-encoding@0.1.0` | `d_model` | integer, `512` | minimum 1; `top` | Width of the sequence representation to which positional information is added. |
| `core.positional-encoding@0.1.0` | `max_len` | integer, `5000` | minimum 1; `bottom` | Maximum sequence length for which positions are prepared. |
| `core.relu@0.1.0` | `inplace` | Boolean, `false` | — | When enabled, applies ReLU in-place to the input tensor. |
| `core.repeat@0.1.0` | `times` | integer, not declared | minimum 1; `top` | Number of independent subflow copies applied in sequence. |
| `core.scale@0.1.0` | `factor` | number, `1.0` | `bottom` | Multiplier applied to every tensor value. |
| `core.softmax@0.1.0` | `dim` | integer, `-1` | `bottom` | Axis along which values are normalized to probabilities. `-1` means the final dimension. |
| `core.subflow-proxy@0.1.0` | — | — | — | No parameters. Delegates analysis and execution to one instance of the nested subflow. |
| `core.transpose@0.1.0` | `dim0` | integer, `-2` | `top` | First dimension to swap; `-2` means the penultimate dimension. |
| `core.transpose@0.1.0` | `dim1` | integer, `-1` | `bottom` | Second dimension to swap; `-1` means the final dimension. |

Dtype choices are specific to each parameter. For example, `Embedding`'s `input_dtype` accepts only integer indices `int32` and `int64`; the vector dtype is limited to the four listed floating-point dtypes. Package constraints are checked when a field is confirmed and by the model validator.

## Resources in the mini LLM project

When you create `New mini LLM`, the project receives editable copies of the graph, dataset and its packages. The palette shows core packages and custom packages declared by the project; their names are descriptions, while `Project resources` and tooltips show exact identities. The custom resources below are all version `1.0.0`.

| Identity | Description and fields | Role in the example |
| --- | --- | --- |
| `llm.causal-mask@1.0.0` | `Causal Mask`: parameter `max_length` (integer, default `128`, minimum 1). Masks future positions in attention scores `[B,T,T]`. | The value limits the context length expected by the package; the node uses the mask in its attention path. Change the parameter in the Inspector. |
| `llm.token-cross-entropy@1.0.0` | `Token Cross Entropy`: no parameters; `loss` output of kind loss; external target named `target` from `batch.targets.target`. Computes mean token cross-entropy from logits `[B,T,V]` to targets `[B,T]`. | Defines the next-token prediction loss. The node's Inspector has no parameter fields; the node can still be selected, connected and removed. |
| `llm.causal-attention@1.0.0` | `Causal Self Attention`: no parameters; `out` output of kind output; legacy description: four-head causal attention, model width 64 and head width 16. | The package remains among project resources but is not used by the updated graph: attention operations are represented by explicit nodes and subflows. The interface has no module for editing a package definition. |
| `llm.tokens@1.0.0` (dataset) | `Autoregressive token sequences`: input `tokens`, dtype `int64`, shape `[B,128]`; target `target`, dtype `int64`, shape `[B,128]`. | The slots can be selected as the active dataset. `Dataset… > Edit` changes name, description and slots; ID/version are fixed in the form, and the interface does not edit adapter code or samples. |

The interface can create new stereotypes with `Create stereotype`, but cannot edit or delete an existing stereotype definition. You can edit node-instance parameters in the Inspector, connect/remove/move nodes and arrange the graph. Among the three LLM resources above, only `Causal Mask` has node parameters that can be changed. The dataset manager does not edit the Python implementation or data files.

Package JSON describes shapes and dtypes for native analysis; the presence of an executable Python package in the example does not mean the Qt client runs Python while editing the graph.
