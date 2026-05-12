from langgraph.graph import StateGraph

from state.agent_state import AgentState


def main():

    def greeting_node(state: AgentState) -> AgentState:
        """Greeting node that processes the message in the state and returns a greeting.

        Args:
            state (AgentState): state containing the message to be processed

        Returns:
            AgentState: the state after processing the greeting
        """
        state["message"] = (
            f"Hello, {state['message']}!, Welcome to the LangGraph tutorial."
        )
        return state

    graph = StateGraph(AgentState)

    graph.add_node("greeter", greeting_node)

    graph.set_entry_point("greeter")
    graph.set_finish_point("greeter")

    app = graph.compile()

    app.get_graph().draw_mermaid_png(output_file_path="./tutorial_1.png")

    result = app.invoke({"message": "Alex"})

    print(result)
    print(result["message"])


if __name__ == "__main__":
    main()
