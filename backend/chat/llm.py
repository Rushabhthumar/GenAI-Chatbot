# llm.py - small LangChain helpers shared by the chat steps.
#
# THE KEY LANGCHAIN IDEA - "chains" built with the | (pipe) operator, called LCEL (LangChain Expression Language):
#     chain = prompt | model | parser
#     chain.invoke({...})   -> runs prompt, sends it to the model, parses the reply, returns the final result
#     chain.stream({...})   -> same, but yields the result piece by piece
# Every piece (prompt, model, parser, retriever) is a "Runnable" with the same .invoke()/.stream() methods,
# so they snap together like RxJS operators in .pipe().

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # Typed chat messages.
from langchain_core.prompts import ChatPromptTemplate  # A prompt with {placeholders} filled at run time.
from langchain_core.runnables import Runnable  # The common interface of everything chainable.
from openai import BadRequestError  # Raised (through LangChain) when Azure's content filter blocks a prompt.
from pydantic import BaseModel  # For the type hint of structured outputs.

from backend.clients import STRUCTURED_OUTPUT_METHOD, get_chat_model  # Shared chat model + how to force structured output.


def structured_chain(system_prompt: str, output_model: type[BaseModel], local_model: str | None = None) -> Runnable:  # prompt | model -> Pydantic object.
    prompt = ChatPromptTemplate.from_messages([  # The prompt template:
        ("system", system_prompt),  # Instructions. NOTE: { and } in a template mean placeholders - don't put raw JSON here.
        ("human", "{input}"),  # The text to classify/route, filled in by .invoke({"input": ...}).
    ])
    model = get_chat_model(0.0, local_model).with_structured_output(  # Wrap the model so it MUST return an `output_model` object.
        output_model,  # The Pydantic class: its fields + descriptions become the schema the model fills in.
        method=STRUCTURED_OUTPUT_METHOD,  # Azure: tool calling under the hood. Ollama: its native JSON-schema mode.
    )
    return prompt | model  # The chain. .invoke({"input": "..."}) returns e.g. a GuardrailResult instance.


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:  # Our {"role","content"} dicts -> LangChain messages.
    return [  # user -> HumanMessage, assistant -> AIMessage.
        HumanMessage(content=turn["content"]) if turn["role"] == "user" else AIMessage(content=turn["content"])
        for turn in history
    ]


def is_content_filter_error(error: Exception) -> bool:  # Did Azure's built-in content filter block this?
    return isinstance(error, BadRequestError) and getattr(error, "code", None) == "content_filter"  # Azure uses code "content_filter".
