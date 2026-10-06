# Shape a network

A neural network begins long before the first weight update. It begins with a question: what enters the model, which transformations does it pass through, and what result should it produce? NNModelling makes that path visible. Each operator becomes a node, each connection shows how a tensor moves, and each subgraph lets you group part of the project and examine it closely.

The canvas is a workbench. Start with an empty project, choose a component from the palette, connect it to others, and change its parameters in the Inspector. Or open an example and follow its reasoning: from an image to the ten logits of an MNIST classifier, or from characters to the predictions of a small language model. Tensor shapes help you read the project: `[B, 128, 64]` means a batch of sequences with 128 positions and 64 features per position. `B` leaves the batch dimension open; the other numbers describe concrete network choices.

This view becomes more useful as a model grows. A wrong connection can be hard to spot in a long definition; here it has a source and a destination. An inconsistent parameter appears beside the node that uses it. A subflow groups a block without hiding its contents: expand its preview or enter its scope to work with individual operators. In the tiny LLM, for example, attention remains readable as a sequence of Query, Key and Value projections, multiplications, a causal mask and Softmax.

## From diagram to experiment

NNModelling divides the work between two complementary tools. The **client** is the desktop application for building and inspecting a project. It checks connections, parameters and tensor shapes, and stores the graph and resources in the project folder. This analysis is useful even while a network is incomplete: you can keep designing, save, and fix the items listed in `Model problems`.

The **server**, also called the backend, receives a copy of the project and trains it in a local container. The dataset supplies inputs and targets; Python operators perform the numerical computation; the optimizer updates the weights. In the client, you can follow training and validation loss, find experiments in the history, and download results. A graph edit made after submission prepares a future experiment: the running job keeps its own immutable copy.

At the end, you can take the model out of the editor. The **weights** are a `safetensors` file; the **wheel** is a Python package containing the graph, resources needed for inference, the dataset adapter and trained weights. Once installed with its dependencies, it exposes `Model.infer(...)` and `Model.inference(...)`: it accepts a value such as a string, converts it into network inputs and decodes the prediction. The backend can then be shut down.

## Terms used in this guide

| Term | Meaning in the software |
| --- | --- |
| **Project** | The folder containing `model.json`, custom packages and declared datasets. Copy the whole folder to preserve an editable project. |
| **Package / stereotype** | A component available in the palette. It defines the node kind, parameters, handles and rules. Core packages ship with the program; custom packages belong to the project. |
| **Node and handle** | An instance of a component and the points through which tensors enter or leave it. An occupied input handle cannot accept a second edge. |
| **Scope / subflow** | A graph scope and the node that contains its subgraph. `Root` is the top level. |
| **Active dataset** | The resource that declares inputs and targets and supplies batches for training, validation and testing. A root Input's `binding` parameter selects the slot name. |
| **Output** | The prediction terminal, drawn as a brown circle. |
| **Loss Output** | The objective terminal, drawn as a red circle. The loss must be computed and connected explicitly. |
| **Job / snapshot** | An experiment and the copy of files and configuration submitted with it. Restoring a snapshot creates a new project. |
| **Epoch / step** | One pass over the training data / one optimizer update. A single epoch can contain many steps. |

## A good first path

To get familiar with the application, open a copy of `mini LLM`, inspect the root graph and select a few nodes. Follow the shapes in the Inspector, enter the decoder block and then its attention scope. Once the path is clear, start the backend and submit the short training run in the [tutorial](tutorial-tiny-llm.md). In a few steps you will have gone through the full cycle: project, checks, experiment and exported model.

The [client guide](client.md) walks through each area of the window; the [server guide](server.md) covers configuration and commands; the [parameter catalog](parameters.md) explains individual fields. Read the pages in order or keep them open beside the software as you work.

This edition describes the Qt interface verified on **6 October 2026**. Screenshots come from the real application; arrows and numbers are editorial annotations. English labels match the buttons exactly. The language-model walkthrough uses `examples/models/tiny-decoder-llm`; other experimental models are outside the tutorial's scope.
