.. Intel® NPU Acceleration Library documentation master file, created by
   sphinx-quickstart on Wed Feb  7 11:48:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Intel® NPU Acceleration Library's documentation!
=====================================

The Intel® NPU Acceleration Library is a Python library designed to boost the efficiency of your applications by leveraging the power of the Intel Neural Processing Unit (NPU) to perform high-speed computations on compatible hardware.

Installation
-------------

Check that your system has an available NPU (`how-to <https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html>`_).

You can install the packet in your machine with

.. code-block:: bash

   pip install intel-npu-acceleration-library


Run a LLaMA model on the NPU
----------------------------

To run LLM models you need to install the `transformers` library


.. code-block:: bash

   pip install transformers

You are now up and running! You can create a simple script like the following one to run a LLM on the NPU


.. code-block:: python
   :emphasize-lines: 2, 7

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


Take note that you only need to use `intel_npu_acceleration_library.compile` to offload the heavy computation to the NPU.

Feel free to check `Usage <usage.html>`_ and `LLM <llm.html>`_ and the `examples <https://github.com/intel/intel-npu-acceleration-library/tree/main/examples>`_ folder for additional use-cases and examples.



Site map
----------------------------

.. toctree::
   Quickstart <self>
   NPU overview <npu.md>
   usage.md
   setup.md
   :maxdepth: 1
   :caption: Library overview:


.. toctree::
   llm.md
   llm_performance.md
   :maxdepth: 1
   :caption: Applications:



.. toctree::
   developer.md
   :maxdepth: 1
   :caption: Developements guide:



.. toctree::
   Python API Reference <python/intel_npu_acceleration_library.rst>
   cpp_reference.rst
   :maxdepth: 1
   :caption: API Reference:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
