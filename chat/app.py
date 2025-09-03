from dotenv import load_dotenv
load_dotenv()

import os
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# ImageTextToTextPipeline, TextGenerationPipeline
# note: could use bitsandbytes for quantization, but won't work on mac

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langsmith import tracing_context

from logging import getLogger

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, pipeline, ImageTextToTextPipeline, TextGenerationPipeline
from langchain_huggingface.llms import HuggingFacePipeline

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

# get typeddict
from typing import TypedDict, List, Dict, Annotated

logger = getLogger(__name__)

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")


def load_mediphi():
    model_id = "microsoft/MediPhi-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2",  # or "sdpa", "xformers" etc.
        # device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
        "eos_token_id": [32000, 32001, 32007],
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_args
    )

    return pipe


def load_medgemma_27b():
    model_id = "google/medgemma-27b-it"

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        # device="cuda",
        use_fast_tokenizer=True,
        use_fast=True
    )

    return pipe


def load_smolvlm():
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        max_new_tokens=512,
        # device="cuda",
    )

    return pipe


def load_model():

    # return load_mediphi()
    return load_medgemma_27b()
    # return load_smolvlm()

model = load_model()


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

def get_messages_length(state: MessagesState):
    """Function to get the length of all messages contatenated."""
    return len("".join(msg.content for msg in state["messages"]))


def format_messages_hf(state: MessagesState, human_prefix: str = "user", ai_prefix: str = "assistant", multimodal: bool = True) -> str:
    messages = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            role = human_prefix
        elif isinstance(message, AIMessage):
            role = ai_prefix
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        content = []
        if multimodal:
            for entry in message.content:
                if isinstance(entry, dict):
                    if entry.get("type") == "text":
                        content.append(entry)
                    elif entry.get("type") == "image":
                        content.append(entry)
                    else:
                        raise ValueError(f"Unsupported content type: {entry.get('type')}")   
        else:
            content = message.content[0]['text']
        message_dict = {
            "role": role,
            "content": content
        }
        messages.append(message_dict)
    return messages


def convert_response_to_aimessage(response: list) -> AIMessage:
    """Convert a response from the model to an AIMessage."""
    if not response or len(response) != 1:
        raise ValueError("Response must contain exactly one message.")
    
    content = response[-1].get("generated_text", "")
    if not content:
        raise ValueError("Response message must contain generated text.")
    
    message_content = [{
        "type": "text",
        "text": content
    }]

    return AIMessage(content=message_content)


# Define the function that calls the model
def call_model(state: MessagesState):
    logger.info(f"Calling model with {len(state['messages'])} messages.")
    logger.info(f"Messages: {state['messages']}")
    # first message
    logger.info(f"First message: {state['messages'][0] if state['messages'] else 'No messages'}")

    # breakpoint()
    messages_hf = format_messages_hf(state)
    logger.info(f"Messages in HF format: {messages_hf}")
    if isinstance(model, ImageTextToTextPipeline):
        response = model(text=format_messages_hf(state), return_full_text=False, max_new_tokens=512)
    elif isinstance(model, TextGenerationPipeline):
        response = model(text_inputs=format_messages_hf(state, multimodal=False), return_full_text=False, max_new_tokens=512)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    logger.info(f"Response from model: {response}")
    
    aimessage = convert_response_to_aimessage(response)
    logger.info(f"Response from model: {aimessage.content}")
    return {"messages": aimessage}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_edge("model", END)

# Add memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)