from typing import TypedDict


class AgentState(TypedDict):
    message: str


class AgentState1(AgentState):
    values: list[int]
    name: str
    result: str


class AgentState2(AgentState):
    name: str
    age: int
    result: str


class AgentState3(AgentState):
    operator_1: int
    operator_2: int
    operation: str
    answer: int


class AgentState4(AgentState):
    name: str
    greeting: str
    numbers: list[int] = []
    counter: int = 0
