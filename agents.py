from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict, Sequence, Generator
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from functools import partial

from prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    SPECIFICATION_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    DEBUGGER_SYSTEM_PROMPT,
)

load_dotenv()

MODEL_ID = "anthropic:claude-sonnet-4-6"
TEMPERATURE = 1.0
WORKERS = ["researcher", "coder", "analyst", "specification", "tester", "debugger"]

MODEL_CATALOG = [
    # Anthropic
    ("Claude Opus 4.6", "anthropic:claude-opus-4-6", "ANTHROPIC_API_KEY"),
    ("Claude Sonnet 4.6", "anthropic:claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ("Claude Haiku 4.5", "anthropic:claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    # OpenAI
    ("GPT-5.4", "openai:gpt-5.4", "OPENAI_API_KEY"),
    ("GPT-5.4 Mini", "openai:gpt-5.4-mini", "OPENAI_API_KEY"),
    ("GPT-5.4 Nano", "openai:gpt-5.4-nano", "OPENAI_API_KEY"),
    ("GPT-5", "openai:gpt-5", "OPENAI_API_KEY"),
    ("GPT-5 Mini", "openai:gpt-5-mini", "OPENAI_API_KEY"),
    ("GPT-4o", "openai:gpt-4o", "OPENAI_API_KEY"),
    ("GPT-4.1", "openai:gpt-4.1", "OPENAI_API_KEY"),
    ("GPT-4.1 Mini", "openai:gpt-4.1-mini", "OPENAI_API_KEY"),
]


class AgentState(TypedDict):
    """Shared state that flows through the entire graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""

    next: Literal[
        "researcher",
        "coder",
        "analyst",
        "specification",
        "tester",
        "debugger",
        "FINISH",
    ] = Field(
        description="Which worker to route to next, or FINISH if the task is complete."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )


class AgentEvent(TypedDict):
    """A single event yielded by the graph run for the TUI to consume."""

    node: str
    content: str
    event_type: str  # "routing", "response", "tool", "done"
    message: BaseMessage | None


@tool
def save_file(content: str, file_name: str) -> bool:
    """Tool for saving any content to a file"""
    try:
        with open(file_name, "w") as file:
            file.write(content)
            return True
    except Exception:
        return False


def supervisor(state: AgentState, model: BaseChatModel) -> AgentState:
    """Supervisor node that decides the agent to route to."""
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


def worker(
    state: AgentState, model: BaseChatModel, system_prompt: str, name: str
) -> AgentState:
    """Specialist worker node that has a specific task."""
    # Exclude supervisor routing messages: they end with an AIMessage, which the
    # Anthropic API rejects when it's not the final turn of a prefilled assistant turn.
    filtered = [m for m in state["messages"] if not (isinstance(m, AIMessage) and getattr(m, "name", None) == "supervisor")]
    messages = [SystemMessage(content=system_prompt)] + filtered
    response = model.invoke(messages)
    response.name = name

    return {"messages": [response]}


def route_after_worker(state: AgentState) -> str:
    """Route to the tool node if the last message has tool calls, otherwise end and wait for user."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "save"

    return END


def route_after_supervisor(state: AgentState) -> str:
    """Conditional edge: where to go after the supervisor decides."""
    if state.get("task_complete"):
        return END

    return state["next_agent"]


def extract_text(content: str | list) -> str:
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return content


def run(
    messages: list[BaseMessage], model: BaseChatModel
) -> Generator[AgentEvent, None, None]:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", partial(supervisor, model=model))
    graph.add_node(
        "researcher",
        partial(
            worker,
            model=model,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            name="researcher",
        ),
    )
    graph.add_node(
        "coder",
        partial(
            worker,
            model=model,
            system_prompt=CODER_SYSTEM_PROMPT,
            name="coder",
        ),
    )
    graph.add_node(
        "analyst",
        partial(
            worker,
            model=model,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            name="analyst",
        ),
    )
    graph.add_node(
        "specification",
        partial(
            worker,
            model=model,
            system_prompt=SPECIFICATION_SYSTEM_PROMPT,
            name="specification",
        ),
    )
    graph.add_node(
        "tester",
        partial(
            worker,
            model=model,
            system_prompt=TESTER_SYSTEM_PROMPT,
            name="tester",
        ),
    )
    graph.add_node(
        "debugger",
        partial(
            worker,
            model=model,
            system_prompt=DEBUGGER_SYSTEM_PROMPT,
            name="debugger",
        ),
    )
    graph.add_node("save", ToolNode(tools=tools))

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {worker: worker for worker in WORKERS} | {END: END},
    )

    for worker_name in WORKERS:
        graph.add_conditional_edges(
            worker_name,
            route_after_worker,
            {"save": "save", END: END},
        )

    # After tool use, end the graph and return to user
    graph.add_edge("save", END)

    app = graph.compile()

    initial_state = {
        "messages": messages,
        "next_agent": "",
        "task_complete": False,
    }

    for step in app.stream(initial_state):
        for node_name, state_update in step.items():
            if "messages" not in state_update:
                continue

            for message in state_update["messages"]:
                if node_name == "supervisor":
                    event_type = "routing"
                elif node_name == "save":
                    event_type = "tool"
                else:
                    event_type = "response"

                text = extract_text(message.content)
                if not text:
                    continue

                yield AgentEvent(
                    node=node_name,
                    content=text,
                    event_type=event_type,
                    message=message,
                )

    yield AgentEvent(
        node="system", content="Task complete.", event_type="done", message=None
    )


def create_model(model_id: str) -> BaseChatModel:
    return init_chat_model(model=model_id, temperature=TEMPERATURE).bind_tools(
        tools=tools
    )


tools = [save_file]
model = init_chat_model(model=MODEL_ID, temperature=TEMPERATURE).bind_tools(tools=tools)
