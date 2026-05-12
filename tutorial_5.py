import random

from langgraph.graph import END, START, StateGraph

from state.agent_state import AgentState4


def main():

    def greeting_node(state: AgentState4) -> AgentState4:
        """Greets User"""

        state["greeting"] = f"Hi there, {state['name']}."
        # Robust so counter always starts from 0.
        state["counter"] = 0
        return state

    def random_node(state: AgentState4) -> AgentState4:
        """Appends random number to the state list"""

        state["numbers"].append(random.randint(0, 10))
        state["counter"] += 1

        return state

    def should_continue(state: AgentState4) -> str:
        """tells if we should continue looping or end"""

        if state["counter"] < 5:
            return "loop"
        else:
            return "exit"

    graph = StateGraph(AgentState4)

    graph.add_node("greeter", greeting_node)
    graph.add_node("random_number", random_node)

    graph.add_edge(START, "greeter")
    graph.add_edge("greeter", "random_number")

    graph.add_conditional_edges(
        "random_number", should_continue, {"loop": "random_number", "exit": END}
    )

    app = graph.compile()

    app.get_graph().draw_mermaid_png(output_file_path="./tutorial_5.png")

    result = app.invoke({"name": "Alex Mercer", "numbers": []})
    print(result)


if __name__ == "__main__":
    main()
