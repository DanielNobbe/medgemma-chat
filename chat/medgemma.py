
import os
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# note: could use bitsandbytes for quantization, but won't work on mac

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langsmith import tracing_context

from logging import getLogger

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import chainlit as cl

from app import graph

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# In this file, we do end-to-end medgemma multimodal integration, since it requires parsing the image and text inputs correctly
# as such, we don't use langgraph here, but it should be straightforward to integrate it later
# actually, using ImagePromptTemplate may allow for both text and image inputs to be handled by the same langchain model
# but the documentation is really bad?

logger = getLogger(__name__)

def load_medgemma_27b():
    model_id = "google/medgemma-27b-it"

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_fast_tokenizer=True,
    )

    return pipe


# TODO: define as override for huggingface pipeline of huggingfacechat in langchain


# for now, just implement simple class with an call_model method

class MedGemmaChat:
    def __init__(self):
        self.pipe = load_medgemma_27b()

    

    def call_model(self, messages: list):
        # Convert messages to the format expected by the model
        # inputs = self.pipe.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # outputs = self.pipe.model.generate(**inputs)
        # response = self.pipe.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response