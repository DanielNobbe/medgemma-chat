import chainlit as cl

from app import graph

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from logging import getLogger

from json import dumps

logger = getLogger(__name__)

"""
message structure for image + text:
{
    'language': None,
    'content': 'hello, this is a message',
    'id': 'ef60cd83-f81f-4ff9-aa33-4d824f0daf87',
    'created_at': '2025-07-23T16:00:41.441332Z',
    'metadata': {'location': 'http://localhost:8000/'},
    'tags': None,
    'author': 'User',
    'type': 'user_message',
    'actions': [],
    'elements': [
        Image(thread_id='47ac2309-85ad-41ba-85d6-5290efc27a15',
              name='001_in_slice_6.png',
              id='3bc88658-6903-4d2d-b5ef-78cde66fee1c',
              chainlit_key='3bc88658-6903-4d2d-b5ef-78cde66fee1c',
              url=None,
              object_key=None,
              path='/Users/daniel/projects/medgemma-chat/.files/c268655a-731a-43a0-8fcb-ba8bb573a64a/3bc88658-6903-4d2d-b5ef-78cde66fee1c.png',
              content=None,
              display='inline',
              size='medium',
              for_id='ef60cd83-f81f-4ff9-aa33-4d824f0daf87',
              language=None,
              mime='image/png')
              ],
    'thread_id': '47ac2309-85ad-41ba-85d6-5290efc27a15'
}

So image is in elements, text in content.

for two images:
{   
    'language': None,
    'content': 'Hello, this is the msg',
    'id': '4827725f-1b24-419e-afe9-0af00c618c18',
    'created_at': '2025-07-23T16:05:08.188368Z',
    'metadata': {'location': 'http://localhost:8000/'},
    'tags': None,
    'author': 'User',
    'type': 'user_message',
    'actions': [],
    'elements': [
        Image(thread_id='b6078dbe-d8a6-4c04-91a2-7dd32f344cb4',
              name='001_in_slice_5.png',
              id='741f1cd8-457d-484c-ab00-87f5c5a07c24',
              chainlit_key='741f1cd8-457d-484c-ab00-87f5c5a07c24',
              url=None,
              object_key=None,
              path='/Users/daniel/projects/medgemma-chat/.files/61a37f9b-a298-42da-9d04-dbe66f644fb7/741f1cd8-457d-484c-ab00-87f5c5a07c24.png',
              content=None,
              display='inline',
              size='medium',
              for_id='4827725f-1b24-419e-afe9-0af00c618c18',
              language=None, mime='image/png'),
        Image(thread_id='b6078dbe-d8a6-4c04-91a2-7dd32f344cb4',
              name='001_in_slice_6.png',
              id='c775774f-8806-4d82-bb41-a2969eff2037',
              chainlit_key='c775774f-8806-4d82-bb41-a2969eff2037',
              url=None,
              object_key=None,
              path='/Users/daniel/projects/medgemma-chat/.files/61a37f9b-a298-42da-9d04-dbe66f644fb7/c775774f-8806-4d82-bb41-a2969eff2037.png',
              content=None,
              display='inline',
              size='medium',
              for_id='4827725f-1b24-419e-afe9-0af00c618c18',
              language=None,
              mime='image/png')
    ], 
    'thread_id': 'b6078dbe-d8a6-4c04-91a2-7dd32f344cb4'
}

So for multiple images, the elements list is just expanded.

Now how should we feed this image into the model?
The images can be fed into the pipeline as part of the message content (`text` field), but it can also be fed as separate input into the `images` key.

Into the images key, we can feed:
A string containing a HTTP(s) link pointing to an image
A string containing a local path to an image
An image loaded in PIL directly

In the messages structure, we can pass an image by url, with the `url` key. Not sure how to pass a local path
or a PIL image.
In the Chainlit Image object, if the image is loaded as bytes, it will have a content field with the bytes. 
In the UI, attaching an image will save it to the local filesystem though, so not sure if we need to support the bytes.

Note that the chainlit UI does not allow messages without text, so we will always have a text message.

"""
def convert_message_to_vlm_format(msg: cl.Message) -> dict:
    """Convert a Chainlit message to a VLM compatible HumanMessage."""
    content = []
    
    # If there are elements, we assume they are images
    if msg.elements:
        for element in msg.elements:
            if isinstance(element, cl.Image):
                if element.url:
                    content.append({"type": "image", "url": element.url})
                elif element.path:
                    content.append({"type": "image", "image": element.path})
                else:
                    raise ValueError(f"Image element must have either a URL or a path. {'The Image does have a non-empty content field, please update this library to support loading the image from bytes.' if element.content else 'It seems the Image element is empty.'}")
            else:
                raise ValueError(f"Unsupported element type: {type(element)}")
    
    # If there's text content, add it as well
    if msg.content:
        content.append({"type": "text", "text": msg.content})
    else:
        raise ValueError("Chainlit message must contain message content. This function needs to be updated if the API has changed.")

    return {
        "role": "user",
        "content": content,
    } # note that this will be converted to a HumanMessage by LangGraph later on, so we don't need to do that here.


def get_response_text(message):
    """Extract the text content from a message."""
    if len(message.content) > 1:
        raise ValueError("Message content should only contain one entry.")
    return message.content[0]['text']

# Note, we are now using LangGraph to handle the chat state.
# which may be beneficial to make sure they don't get out of sync.
# @cl.on_message
# async def on_message(msg: cl.Message):
#     config = {"configurable": {"thread_id": cl.context.session.id}}
#     # cb = cl.LangchainCallbackHandler()
#     final_answer = cl.Message(content="")
#     # breakpoint()

#     logger.info(f"Received message: {vars(msg)}")

#     logger.info(f"Converting message to VLM format: {convert_message_to_vlm_format(msg)}")

#     for msg, metadata in graph.stream({"messages": [convert_message_to_vlm_format(msg)]}, stream_mode="messages", config=RunnableConfig(callbacks=[], **config)):
#         logger.debug(f"Processing message: {msg.content} with metadata: {metadata}")
#         if (
#             msg.content
#             and not isinstance(msg, HumanMessage)
#             and metadata["langgraph_node"] == "model"  # NOTE: This may be different based on graph structure
#         ):
#             logger.info(f"Streaming message: {msg.content}")

#             await final_answer.stream_token(get_response_text(msg))

#     await final_answer.send()

# synchronous version
@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    # cb = cl.LangchainCallbackHandler()

    response = graph.invoke({"messages": [convert_message_to_vlm_format(msg)]}, config=RunnableConfig(callbacks=[], **config))['messages'][-1]

    logger.info(f"Graph response: {response}")

    final_answer = cl.Message(content=get_response_text(response))
    # breakpoint()

    # logger.info(f"Received message: {vars(msg)}")

    # logger.info(f"Converting message to VLM format: {convert_message_to_vlm_format(msg)}")

    # for msg, metadata in graph.stream({"messages": [convert_message_to_vlm_format(msg)]}, stream_mode="messages", config=RunnableConfig(callbacks=[], **config)):
    #     logger.debug(f"Processing message: {msg.content} with metadata: {metadata}")
    #     if (
    #         msg.content
    #         and not isinstance(msg, HumanMessage)
    #         and metadata["langgraph_node"] == "model"  # NOTE: This may be different based on graph structure
    #     ):
    #         logger.info(f"Streaming message: {msg.content}")

    #         await final_answer.stream_token(get_response_text(msg))

    await final_answer.send()

    return final_answer
