import asyncio
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the content of document with provided string

    Args:
        content (str): string that we will update the document with

    Returns:
        str: a genric message
    """
    global document_content
    document_content = content
    return (
        f"Document has been updated successfully, Document Content ; {document_content}"
    )


@tool
def save(filename: str) -> str:
    """Save the final document once approved

    Args:
        filename (str): name of the file where the content will be stored

    Returns:
        str: a generic message after saving
    """

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    global document_content

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return "File has been saved successfully"
    except Exception as err:
        return f"Error Saving Document, {err}"


tools = [update, save]
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite").bind_tools(tools=tools)


async def llm_call(state: AgentState) -> AgentState:
    SYSTEM_PROMPT = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_message = HumanMessage(content="Hello")
    else:
        user_query = input("\nWhat would you like to change in the document? ")
        print(f"\nUSER : {user_query}")
        user_message = HumanMessage(content=user_query)

    all_messages = [SYSTEM_PROMPT] + list(state["messages"]) + [user_message]

    llm_response = await llm.ainvoke(all_messages)

    print(f"\nAI Responde : {llm_response.content}")

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        print(f"\nUsing Tools : {[tc for tc in llm_response.tool_calls]}")

    return {"messages": [user_message, llm_response]}


def should_continue(state: AgentState) -> str:
    """Determines to continue or end the loop"""

    if not state["messages"]:
        return "continue"

    for message in reversed(state["messages"]):
        print("message\n : ", message)
        if (
            isinstance(message, ToolMessage)
            and "save" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"

    return "continue"


def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool"
    return "llm_node"  # ← loop back for user input instead of ending


def print_messages(messages):
    if not messages:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"\nTool Result : {message.content}")


graph = StateGraph(AgentState)

graph.add_node("llm_node", llm_call)
graph.add_node("tool", ToolNode(tools))

graph.add_edge(START, "llm_node")

graph.add_conditional_edges(
    "llm_node", should_use_tool, {"tool": "tool", "llm_node": "llm_node"}
)

graph.add_conditional_edges(
    "tool", should_continue, {"continue": "llm_node", "end": END}
)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="./tutorial_9.png")


async def run_agent():
    print("=======Drafter========")

    state = {"messages": []}

    async for step in app.astream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])


if __name__ == "__main__":
    asyncio.run(run_agent())
