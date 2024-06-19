# Intel® NPU Acceleration Library

[![Test](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/test.yml/badge.svg)](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/test.yml) [![Style](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/style.yml/badge.svg)](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/style.yml) [![Documentation](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/documentation.yml/badge.svg)](https://github.com/intel/intel-npu-acceleration-library/actions/workflows/documentation.yml)

[![PyPI version](https://badge.fury.io/py/intel-npu-acceleration-library.svg)](https://badge.fury.io/py/intel-npu-acceleration-library) [![Downloads](https://static.pepy.tech/badge/intel-npu-acceleration-library)](https://pepy.tech/project/intel-npu-acceleration-library)

[Documentation](https://intel.github.io/intel-npu-acceleration-library/)

The Intel® NPU Acceleration Library is a Python library designed to boost the efficiency of your applications by leveraging the power of the Intel Neural Processing Unit (NPU) to perform high-speed computations on compatible hardware.

_Note: The **Intel® NPU Acceleration Library** is currently in active development, with our team  working to introduce a variety of features that are anticipated to dramatically enhance performance._

## Intel NPU

The Intel NPU is an AI accelerator integrated into Intel Core Ultra processors, characterized by a unique architecture comprising compute acceleration and data transfer capabilities. Its compute acceleration is facilitated by Neural Compute Engines, which consist of hardware acceleration blocks for AI operations like Matrix Multiplication and Convolution, alongside Streaming Hybrid Architecture Vector Engines for general computing tasks.

To optimize performance, the NPU features DMA engines for efficient data transfers between system memory and a managed cache, supported by device MMU and IOMMU for security isolation. The NPU's software utilizes compiler technology to optimize AI workloads by directing compute and data flow in a tiled fashion, maximizing compute utilization primarily from scratchpad SRAM while minimizing data transfers between SRAM and DRAM for optimal performance and power efficiency.

Some useful links

- Intel AI PC ([link](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html?wapkw=NPU))
- Intel Core Ultra Processor line ([link](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/core-ultra-series-1-product-brief.html?wapkw=NPU))
- AI Acceleration and NPU explained ([video](https://www.youtube.com/watch?v=QSzNoX0qplE))

## Feature roadmap

In our quest to significantly improve the library's performance, we are directing our efforts toward implementing a range of key features, including:

- [x] **8-bit quantization**
- [x] **4-bit Quantization and GPTQ**
- [x] **NPU-Native mixed precision inference**
- [x] **Float16 support**
- [ ] **BFloat16 (Brain Floating Point Format)**
- [x] **`torch.compile` support**
- [x] **LLM MLP horizontal fusion implementation**
- [x] **Static shape inference**
- [x] **MHA NPU inference**
- [ ] **NPU/GPU hetero compute**
- [ ] **Paper**

Make sure to stay updated with the project's progress as these exciting enhancements are on the horizon. External contributions are very welcomed! If you want to participate in this library development, please check the [Contributing](CONTRIBUTING.md) guide, the [developer guide](https://intel.github.io/intel-npu-acceleration-library/developer.html) and the list of open [issues](https://github.com/intel/intel-npu-acceleration-library/issues)

## Setup

Check that your system has an available NPU ([how-to](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)).

You can install the packet in your machine with

```bash
   pip install intel-npu-acceleration-library
```

You can also install the package on Windows and Linux from source by typing

```bash
pip install "intel-npu-acceleration-library @ git+https://github.com/intel/intel-npu-acceleration-library.git"
```

To build the package you need a compiler in your system (Visual Studio 2019 suggested for Windows build). MacOS is not yet supported. At the moment only Ubuntu OS is supported for Linux build. If you need a library for your specific OS, please open an [issue](https://github.com/intel/intel-npu-acceleration-library/issues)

The library is intended to be used with Intel Core Ultra processors, which have an integrated `NPU` (Neural Processing Unit). For best performance please install/update the NPU drivers to the latest version. ([Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html), [Linux](https://github.com/intel/linux-npu-driver)).

## Usage

For implemented examples, please check the `examples` folder

### Run a single MatMul in the NPU

```python
from intel_npu_acceleration_library.backend import MatMul
import numpy as np

inC, outC, batch = ... # Define your own values

# Create both inputs
X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch, profile=False)

result = mm.run(X1, X2)

```

### Compile a model for the NPU

If you have `pytorch`>=2.0.0 installed you can use torch compile to optimize your model for the NPU

```python
import intel_npu_acceleration_library
import torch

# Compile model for the NPU
# model a torch.nn.Module class. Model can be quantized JIT
optimized_model = torch.compile(model, backend="npu")

# Use the model as usual

```

In windows torch.compile is not supported yet. So you might want to use the explicit function `intel_npu_acceleration_library.compile`. This is true also if you use a `pytorch` version < 2.0.0

```python
import intel_npu_acceleration_library
optimized_model = intel_npu_acceleration_library.compile(model, dtype=torch.int8)

# Use the model as usual

```

### Run a Tiny-llama model on the NPU

```python
from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = NPUModelForCausalLM.from_pretrained(model_id, use_cache=True, dtype=torch.int8).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)


query = input("Ask something: ")
prefix = tokenizer(query, return_tensors="pt")["input_ids"]


generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
)

print("Run inference")
_ = model.generate(**generation_kwargs)

```
