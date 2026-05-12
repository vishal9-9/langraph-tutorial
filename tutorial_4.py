from langgraph.graph import StateGraph, START, END
from state.agent_state import AgentState3


def main():
    def add(state: AgentState3) -> AgentState3:
        """Adds 2 number"""
        state["answer"] = state["operator_1"] + state["operator_2"]
        return state

    def subtract(state: AgentState3) -> AgentState3:
        """Subtracts 2 number"""
        state["answer"] = state["operator_1"] - state["operator_2"]
        return state

    def decide_operation_node(state: AgentState3) -> str:
        """Decides which node to call"""
        if state["operation"] == "+":
            return "addition_operation"
        elif state["operation"] == "-":
            return "subtraction_operation"

    graph = StateGraph(AgentState3)

    graph.add_node("add_node", add)
    graph.add_node("subtract_node", subtract)
    graph.add_node("router", lambda state: state)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        decide_operation_node,
        {
            "addition_operation": "add_node",
            "subtraction_operation": "subtract_node",
        },  # Edge(Output of decide_operation_node): Node
    )

    graph.add_edge("add_node", END)
    graph.add_edge("subtract_node", END)

    app = graph.compile()

    app.get_graph().draw_mermaid_png(output_file_path="./tutorial_4.png")

    result = app.invoke({"operator_1": 10, "operator_2": 15, "operation": "+"})

    print(result)
    print(result["answer"])


if __name__ == "__main__":
    main()
