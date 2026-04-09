from typing import Annotated, Literal, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from functools import partial
from dotenv import load_dotenv

from prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
)

load_dotenv()

temperature = 1.0
model = init_chat_model(model="anthropic:claude-haiku-4-5", temperature=temperature)


class AgentState(TypedDict):
    """Shared state that flows through the entire graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["researcher", "coder", "analyst", "FINISH"] = Field(
        description="Which worker to route to next, or FINISH if the task is complete."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )


def supervisor(state: AgentState) -> AgentState:
    """Supervisor node that decides the agent to route too."""
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + list(
        state["messages"]
    )

    structured_output = model.with_structured_output(SupervisorDecision)
    decision = structured_output.invoke(messages)

    return {
        "messages": [
            AIMessage(
                content=f"Routing to **[{decision.next.upper()}]** — {decision.reasoning}",
                name="supervisor",
            )
        ],
        "next_agent": decision.next,
        "task_complete": decision.next == "FINISH",
    }


def worker(state: AgentState, system_prompt: str, name: str) -> AgentState:
    """Specialist worker node that has a specific task"""
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = model.invoke(messages)
    response.name = name

    return {
        "messages": [AIMessage(content=response.content, name=name)],
    }


def route_after_supervisor(state: AgentState) -> str:
    """Conditional edge: where to go after the supervisor decides."""
    if state.get("task_complete"):
        return END

    return state["next_agent"]


def run(query: str) -> list[BaseMessage]:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node(
        "researcher",
        partial(worker, system_prompt=RESEARCHER_SYSTEM_PROMPT, name="researcher"),
    )
    graph.add_node(
        "coder", partial(worker, system_prompt=CODER_SYSTEM_PROMPT, name="coder")
    )
    graph.add_node(
        "analyst", partial(worker, system_prompt=ANALYST_SYSTEM_PROMPT, name="analyst")
    )

    graph.set_entry_point("supervisor")

    # Supervisor conditionally routes to a worker or the end
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "analyst": "analyst",
            END: END,
        },
    )

    # After any worker finishes, route back to the supervisor for the next decision
    for worker_name in ["researcher", "coder", "analyst"]:
        graph.add_edge(worker_name, "supervisor")

    app = graph.compile()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "",
        "task_complete": False,
    }

    print(f"User query: {query}\n")

    all_messages = list(initial_state["messages"])

    for step in app.stream(initial_state):
        for node_name, state_update in step.items():
            if "messages" in state_update:
                all_messages.extend(state_update["messages"])

                for message in state_update["messages"]:
                    print(f"[{node_name.upper()}]: {message.content}\n")

    return all_messages


if __name__ == "__main__":
    # Research task
    run(
        "What are the key differences between GNNs and Graph Transformers? "
        "Summarize the trade-offs in expressivity and scalability."
    )

    # Coding task
    run(
        "Write a Python function that implements the Floyd-Warshall algorithm "
        "with type hints and a docstring."
    )

    # Multi-step task requiring multiple agents
    run(
        "I have a dataset of 10,000 molecules with SMILES strings and solubility labels. "
        "First, explain what featurization strategies I should use. "
        "Then write PyTorch Geometric code for a GIN-based classifier. "
        "Finally, analyze what batch size and learning rate schedule would work best."
    )
