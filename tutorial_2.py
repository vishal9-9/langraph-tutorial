from langgraph.graph import StateGraph

from state.agent_state import AgentState1


def main():

    def process_node(state: AgentState1) -> AgentState1:
        """Node that processes the values in the state and returns a result.

        Args:
            state (AgentState1): state containing the values to be processed

        Returns:
            AgentState1: the state after processing the values
        """

        state["result"] = (
            f"Hello {state['name']}, the sum of the values is {sum(state['values'])}."
        )
        return state

    graph = StateGraph(AgentState1)

    graph.add_node("processor", process_node)

    graph.set_entry_point("processor")
    graph.set_finish_point("processor")

    app = graph.compile()
    
    app.get_graph().draw_mermaid_png(output_file_path="./tutorial_2.png")
    
    result = app.invoke({"name": "Alex", "values": [1, 2, 3, 4, 5]})

    print(result)
    print(result["result"])


if __name__ == "__main__":
    main()
