import asyncio
from typing import Annotated, Sequence, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def addition(a: Union[int, float], b: Union[int, float]):
    """Adds two number and returns them

    Args:
        a (int, float): operator 1
        b (int, float): operator 2
    """
    return a + b


@tool
def subtraction(a: Union[int, float], b: Union[int, float]):
    """subtracts two number and returns them

    Args:
        a (int, float): operator 1
        b (int, float): operator 2
    """
    return a + b


@tool
def multiplication(a: Union[int, float], b: Union[int, float]):
    """multiplies two number and returns them

    Args:
        a (int, float): operator 1
        b (int, float): operator 2
    """
    return a + b


tools = [addition, subtraction, multiplication]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools(tools=tools)


async def llm_call(state: AgentState) -> AgentState:
    SYSTEM_PROMPT = SystemMessage(
        content=(
            "You are a helpful assistant. Answer user queries the best you can. "
            "After using tools, always summarize the results clearly for the user."
        )
    )

    llm_response = await llm.ainvoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [llm_response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_call"
    else:
        return "end"


graph = StateGraph(AgentState)
graph.add_node("llm_node", llm_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)


graph.add_edge(START, "llm_node")

graph.add_conditional_edges(
    "llm_node", should_continue, {"tool_call": "tools", "end": END}
)

graph.add_edge("tools", "llm_node")

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="./tutorial_8.png")


async def main():
    stream = app.astream(
        {"messages": [("user", "Add 3, 4 and multiply 15, 5 and subtract 10, 2")]},
        stream_mode="values",
    )
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
