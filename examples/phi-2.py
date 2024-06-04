#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, TextStreamer
import intel_npu_acceleration_library as npu_lib

model_id = "microsoft/Phi-2"

model = npu_lib.NPUModelForCausalLM.from_pretrained(
    model_id, use_cache=True, dtype=npu_lib.int4
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.2,
    streamer=streamer,
)

local_llm = HuggingFacePipeline(pipeline=pipe)
pipe.model.config.pad_token_id = pipe.model.config.eos_token_id


template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question = "What's the distance between the Earth and the Moon?"

llm_chain.run(question)
