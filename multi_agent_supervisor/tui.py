from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Header, Footer, Static, Input, Label, Rule
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.text import Text

from agents import MODEL_ID, TEMPERATURE, WORKERS, run, model

AGENT_COLORS = {
    "supervisor": "bright_yellow",
    "researcher": "bright_cyan",
    "coder": "bright_green",
    "analyst": "bright_magenta",
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

    def render(self) -> Text:
        text = Text()

        # Model badge
        text.append(" MODEL ", style="bold white on dark_blue")
        text.append(f" {MODEL_ID} ", style="bright_white")
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


class MultiAgentTUI(App):
    TITLE = "Multi-Agent Supervisor"
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
        ("escape", "quit", "Quit"),
    ]

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

        self._status_bar = StatusBar()

        # Welcome message
        log = self.query_one("#message-log", VerticalScroll)
        welcome = Text()
        welcome.append(
            "Welcome to the Multi-Agent Supervisor\n", style="bold bright_white"
        )
        welcome.append(
            "Enter a prompt below to start the multi-agent supervisor.\n",
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
        self._status_bar.active_agent = active_agent
        self._status_bar.is_thinking = is_thinking
        self._status_bar.step_count = step_count
        self.query_one("#status-bar").update(self._status_bar.render())

    @work(thread=True)
    def run_agents(self, query: str) -> None:
        step_count = 0

        self.call_from_thread(self.update_status, "supervisor", True, step_count)

        for event in run(query, model):
            step_count += 1
            node = event["node"]
            content = event["content"]
            event_type = event["event_type"]

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

    def action_clear_log(self) -> None:
        log = self.query_one("#message-log", VerticalScroll)
        log.remove_children()

    def action_quit(self) -> None:
        self.exit()
