import os

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Static, Input, Rule, OptionList
from textual.widgets.option_list import Option
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.text import Text

from langchain_core.messages import HumanMessage

from agents import (
    MODEL_ID,
    TEMPERATURE,
    WORKERS,
    MODEL_CATALOG,
    run,
    model,
    create_model,
)

AGENT_COLORS = {
    "supervisor": "bright_yellow",
    "researcher": "bright_cyan",
    "coder": "bright_green",
    "analyst": "bright_magenta",
    "specification": "bright_blue",
    "tester": "orange3",
    "debugger": "red1",
    "save": "bright_red",
    "system": "dim white",
    "user": "bright_white",
}

EVENT_ICONS = {
    "routing": "->",
    "response": ">>",
    "tool": "!!",
    "done": "OK",
}


class StatusBar(Static):
    """Displays the current agent, model, and status."""

    active_agent: reactive[str] = reactive("idle")
    is_thinking: reactive[bool] = reactive(False)
    step_count: reactive[int] = reactive(0)
    model_id: reactive[str] = reactive(MODEL_ID)

    def render(self) -> Text:
        text = Text()

        # Model badge
        text.append(" MODEL ", style="bold white on dark_blue")
        text.append(f" {self.model_id} ", style="bright_white")
        text.append("  ")

        # Temperature
        text.append(" TEMP ", style="bold white on dark_green")
        text.append(f" {TEMPERATURE} ", style="bright_white")
        text.append("  ")

        # Active agent badge
        agent_color = AGENT_COLORS.get(self.active_agent, "white")
        text.append(" AGENT ", style="bold white on dark_red")
        text.append(f" {self.active_agent.upper()} ", style=f"bold {agent_color}")
        text.append("  ")

        # Step counter
        text.append(" STEPS ", style="bold white on rgb(80,80,80)")
        text.append(f" {self.step_count} ", style="bright_white")
        text.append("  ")

        # Thinking indicator
        if self.is_thinking:
            text.append(" ◉ Thinking... ", style="bold bright_yellow italic")
        else:
            text.append(" ◯ Ready ", style="dim white")

        return text


class MessageBlock(Static):
    """A single message in the conversation log."""

    def __init__(self, node: str, content: str, event_type: str) -> None:
        super().__init__()
        self.node = node
        self.msg_content = content
        self.event_type = event_type

    def compose(self) -> ComposeResult:
        color = AGENT_COLORS.get(self.node, "white")
        icon = EVENT_ICONS.get(self.event_type, "  ")

        yield Static(
            Text.assemble(
                (f" {icon} ", "bold dim"),
                (f"[{self.node.upper()}]", f"bold {color}"),
            ),
            classes="msg-label",
        )

        if self.event_type == "done":
            yield Static(
                Text(f"  {self.msg_content}", style="bold dim green"),
                classes="msg-body",
            )
        else:
            yield Static(Markdown(self.msg_content), classes="msg-body")


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal screen for selecting a model from the catalog."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }

    #model-picker-container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    #model-picker-title {
        text-align: center;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #model-list {
        height: auto;
        max-height: 20;
    }
    """

    def __init__(self, current_model_id: str) -> None:
        super().__init__()
        self.current_model_id = current_model_id

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-container"):
            yield Static(
                Text("Select Model", style="bold bright_white"),
                id="model-picker-title",
            )

            options: list[Option] = []
            current_provider = ""

            for pretty_name, model_id, env_var in MODEL_CATALOG:
                provider = model_id.split(":")[0].upper()
                if provider != current_provider:
                    if current_provider:
                        # Visual separator between providers
                        options.append(
                            Option(
                                Text(f"  ── {provider} ──", style="dim bold"),
                                disabled=True,
                            )
                        )
                    else:
                        options.append(
                            Option(
                                Text(f"  ── {provider} ──", style="dim bold"),
                                disabled=True,
                            )
                        )
                    current_provider = provider

                has_key = bool(os.environ.get(env_var))
                marker = "●" if model_id == self.current_model_id else "○"

                if has_key:
                    label = Text.assemble(
                        (
                            f" {marker} ",
                            (
                                "bold bright_green"
                                if model_id == self.current_model_id
                                else "dim"
                            ),
                        ),
                        (f"{pretty_name}", "bright_white"),
                        ("  ", ""),
                        (f"{model_id}", "dim"),
                    )
                else:
                    label = Text.assemble(
                        (f" ✗ ", "dim red"),
                        (f"{pretty_name}", "dim"),
                        ("  ", ""),
                        (f"{model_id}", "dim"),
                        ("  (no API key)", "dim red italic"),
                    )

                options.append(Option(label, id=model_id, disabled=not has_key))

            yield OptionList(*options, id="model-list")
            yield Static(
                Text("ESC to cancel  •  ENTER to select", style="dim"),
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentForgeTUI(App):
    TITLE = "AgentForge"
    CSS = """
    Screen {
        background: $surface;
    }

    #status-bar {
        dock: top;
        height: 1;
        background: $panel;
        padding: 0 1;
        margin-bottom: 1;
    }

    #message-log {
        height: 1fr;
        padding: 0 1;
        margin: 0 0 1 0;
        scrollbar-size: 1 1;
    }

    MessageBlock {
        margin: 0 0 1 0;
        padding: 0 1;
        layout: vertical;
    }

    .msg-label {
        height: auto;
        margin: 0 0 0 0;
    }

    .msg-body {
        height: auto;
        margin: 0 0 0 5;
    }

    #input-area {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }

    #prompt-input {
        width: 1fr;
    }

    #separator {
        dock: bottom;
        margin: 0 1;
        color: $text-muted;
    }

    #workers-bar {
        dock: top;
        height: 1;
        background: $panel;
        padding: 0 1;
        margin-bottom: 0;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("ctrl+k", "pick_model", "Model"),
        ("escape", "quit", "Quit"),
    ]

    current_model_id: reactive[str] = reactive(MODEL_ID)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="workers-bar")
        yield Static(id="status-bar")
        yield VerticalScroll(id="message-log")
        yield Rule(id="separator")
        yield Input(placeholder="Enter your prompt...", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        workers_bar = self.query_one("#workers-bar", Static)
        text = Text()
        text.append(" WORKERS ", style="bold white on rgb(60,60,60)")
        text.append("  ")

        for worker in WORKERS:
            color = AGENT_COLORS[worker]
            text.append(f" ● {worker.upper()} ", style=f"bold {color}")
            text.append(" ")

        workers_bar.update(text)

        status = StatusBar()
        status.id = "status-bar-widget"
        self.query_one("#status-bar").update(status.render())

        self.status_bar = StatusBar()
        self.model = model
        self.conversation_history: list = []

        # Welcome message
        log = self.query_one("#message-log", VerticalScroll)
        welcome = Text()
        welcome.append("Welcome to AgentForge!\n", style="bold bright_white")
        welcome.append(
            "Enter a prompt below to start AgentForge.\n",
            style="dim white",
        )
        welcome.append(
            f"Model: {MODEL_ID}  |  Workers: {', '.join(WORKERS)}  |  Ctrl+C to quit",
            style="dim",
        )
        log.mount(Static(welcome))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        input_widget = self.query_one("#prompt-input", Input)
        input_widget.value = ""
        input_widget.disabled = True

        self.add_message("user", query, "response")
        self.run_agents(query)

    def add_message(self, node: str, content: str, event_type: str) -> None:
        log = self.query_one("#message-log", VerticalScroll)
        block = MessageBlock(node=node, content=content, event_type=event_type)
        log.mount(block)
        log.scroll_end(animate=False)

    def update_status(
        self,
        active_agent: str = "idle",
        is_thinking: bool = False,
        step_count: int = 0,
    ) -> None:
        self.status_bar.active_agent = active_agent
        self.status_bar.is_thinking = is_thinking
        self.status_bar.step_count = step_count
        self.status_bar.model_id = self.current_model_id
        self.query_one("#status-bar").update(self.status_bar.render())

    @work(thread=True)
    def run_agents(self, query: str) -> None:
        step_count = 0

        self.conversation_history.append(HumanMessage(content=query))

        self.call_from_thread(self.update_status, "supervisor", True, step_count)

        for event in run(list(self.conversation_history), self.model):
            step_count += 1
            node = event["node"]
            content = event["content"]
            event_type = event["event_type"]
            message = event["message"]

            # Accumulate agent messages into history for future runs
            if message is not None:
                self.conversation_history.append(message)

            if event_type == "done":
                self.call_from_thread(self.add_message, node, content, event_type)
                self.call_from_thread(self.update_status, "idle", False, step_count)
            else:
                # Update status to show current agent thinking
                next_thinking = event_type == "routing"
                self.call_from_thread(
                    self.update_status, node, next_thinking, step_count
                )
                self.call_from_thread(self.add_message, node, content, event_type)

                # After routing, the next agent will be thinking
                if event_type == "routing":
                    # Extract target agent from routing message
                    for worker in WORKERS:
                        if worker.upper() in content.upper():
                            self.call_from_thread(
                                self.update_status, worker, True, step_count
                            )

                            break

        # Re-enable input
        def enable_input() -> None:
            input_widget = self.query_one("#prompt-input", Input)
            input_widget.disabled = False
            input_widget.focus()

        self.call_from_thread(enable_input)

    def on_model_selected(self, model_id: str | None) -> None:
        if model_id is None or model_id == self.current_model_id:
            return

        self.model = create_model(model_id)
        self.current_model_id = model_id

        # Find pretty name for the selected model
        pretty_name = model_id
        for name, mid, _ in MODEL_CATALOG:
            if mid == model_id:
                pretty_name = name
                break

        self.add_message(
            "system",
            f"Model switched to **{pretty_name}** (`{model_id}`)",
            "done",
        )
        self.update_status()

    def action_pick_model(self) -> None:
        self.push_screen(
            ModelPickerScreen(self.current_model_id),
            callback=self.on_model_selected,
        )

    def action_clear_log(self) -> None:
        log = self.query_one("#message-log", VerticalScroll)
        log.remove_children()

    def action_quit(self) -> None:
        self.exit()


if __name__ == "__main__":
    app = AgentForgeTUI()
    app.run()
