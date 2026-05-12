from langgraph.graph import StateGraph

from state.agent_state import AgentState2


def main():

    def greeting_node(state: AgentState2) -> AgentState2:
        """Greeting node that processes the name and age in the state and returns a greeting.

        Args:
            state (AgentState2): state containing the name and age to be processed

        Returns:
            AgentState2: the state after processing the greeting
        """

        state["result"] = f"Hello, {state['name']}!"
        return state

    def age_node(state: AgentState2) -> AgentState2:
        """Age node that processes the age in the state and returns a message based on the age.

        Args:
            state (AgentState2): state containing the age to be processed
        Returns:
            AgentState2: the state after processing the age
        """

        if state["age"] < 18:
            state["result"] += " You are a minor."
        else:
            state["result"] += " You are an adult."

        return state

    graph = StateGraph(AgentState2)

    graph.add_node("greeter", greeting_node)
    graph.add_node("age_processor", age_node)

    graph.set_entry_point("greeter")
    graph.add_edge("greeter", "age_processor")
    graph.set_finish_point("age_processor")

    app = graph.compile()

    app.get_graph().draw_mermaid_png(output_file_path="./tutorial_3.png")

    result = app.invoke({"name": "Alex", "age": 25})

    print(result)


if __name__ == "__main__":
    main()
