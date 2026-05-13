import asyncio
from typing import List, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


async def process(state: AgentState) -> AgentState:
    """Basic Agent Bot Node"""

    llm_response = await llm.ainvoke(state["messages"])
    response_content = llm_response.content

    state["messages"].append(AIMessage(content=response_content))
    print(f"\nAI : {response_content}")
    return state


graph = StateGraph(AgentState)

graph.add_node("agent_node", process)

graph.add_edge(START, "agent_node")
graph.add_edge("agent_node", END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="./tutorial_7.png")


async def main():
    conversation = []

    user_query = input("Your Query : ")
    while user_query != "exit":
        conversation.append(HumanMessage(content=user_query))
        await app.ainvoke(AgentState(messages=conversation))
        user_query = input("Your Query : ")


if __name__ == "__main__":
    asyncio.run(main())
