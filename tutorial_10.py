import asyncio
import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph, add_messages

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", output_dimensionality=768
)


pdf_path = "files/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found : {pdf_path}")

pdf_loader = PyPDFLoader(file_path=pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF loaded successfully and has {len(pages)} pages.")
except Exception as e:
    print(f"Error loading PDF, {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

page_split = text_splitter.split_documents(pages)

chroma_directory = r"./db"
collection_name = "stock_market_performance"

try:
    vector_store = Chroma.from_documents(
        documents=page_split,
        embedding=embedding_model,
        persist_directory=chroma_directory,
        collection_name=collection_name,
    )
    print("ChromaDB Vector store created.")
except Exception as err:
    print(f"Error setting up ChromaDB: {str(err)}")
    raise

chunk_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """Returns and return approprite chunks realted to the query specified in parameter from the documents

    Args:
        query (str): used to do a similarity search in vector db

    Returns:
        str: chunks realted to query
    """

    docs = chunk_retriever.invoke(query)

    if not docs:
        return "No Chunks found realted to your query in the document"

    result = []

    for i, doc in enumerate(docs):
        result.append(f"Document {i + 1}: \n {doc.page_content}")

    return "\n\n".join(result)


tools = [retriever_tool]

llm = llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


SYSTEM_PROMPT = """
    You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
    Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {tool.name: tool for tool in tools}


async def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    llm_response = await llm.ainvoke(messages)
    return {"messages": [llm_response]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for tool_call in tool_calls:
        print(f"Calling tool: {tool_call['name']}, with query: {tool_call['args']}")

        if tool_call["name"] not in tools_dict:
            print("\n tool does not exist")
            result = "Incorrect tool name, please retry and select tool from list of Available tools"

        else:
            result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
            print(f"Result : {len(str(result))}")

        results.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=str(result),
            )
        )

    print("Tool call complete. Back to Model.")
    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm_node", call_llm)
graph.add_node("retriever_agent", take_action)


graph.add_edge(START, "llm_node")
graph.add_conditional_edges(
    "llm_node", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm_node")


app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="./tutorial_10.png")


async def main():
    while True:
        user_query = input("User : ")

        conversation = []

        conversation.append(HumanMessage(content=user_query))

        response = await app.ainvoke({"messages": conversation})
        
        print(response["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
