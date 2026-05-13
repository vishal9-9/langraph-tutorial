import asyncio
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


async def process(state: AgentState) -> AgentState:
    ai_response = await llm.ainvoke(state["messages"])
    print(f"\n AI : {ai_response.content}")
    return state


graph = StateGraph(AgentState)

graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="./tutorial_6.png")


async def main():
    user_question = input("Enter Your Query :- ")
    while user_question != "exit":
        await app.ainvoke(AgentState(messages=[HumanMessage(content=user_question)]))
        user_question = input("Enter Your Query :- ")


if __name__ == "__main__":
    asyncio.run(main())
