# AgentForge

AgentForge is a multi-agent orchestration system built with **LangGraph** and **LangChain**, driven from a rich terminal UI powered by **Textual**. A supervisor LLM routes user requests to a team of six specialist agents, each with a focused role, and streams their reasoning, routing decisions, and tool calls back to an interactive TUI in real time.

```
┌─ AgentForge ──────────────────────────────────────────────────────┐
│  WORKERS   ● RESEARCHER  ● CODER  ● ANALYST  ● SPEC  ● TEST  ● DEBUG │
│  MODEL  claude-sonnet-4-6   TEMP 1.0   AGENT  SUPERVISOR   STEPS 4  │
│                                                                     │
│  >> [SUPERVISOR] Routing to [SPECIFICATION] — request is open-ended │
│  >> [SPECIFICATION] ...                                             │
│  >> [CODER] ...                                                     │
│  !! [SAVE] wrote solution.py                                        │
│  OK [SYSTEM] Task complete.                                         │
│                                                                     │
│  > Enter your prompt...                                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Overview

AgentForge is a supervised multi-agent system. A **supervisor** agent inspects the conversation state and decides — via structured output — which specialist should act next. Specialists return their work to the shared message state, and the supervisor either routes to another specialist or finishes the task. Any agent may invoke a `save_file` tool to persist results to disk.

The entire graph executes inside a LangGraph `StateGraph`, and every node transition is streamed as an event into a Textual TUI with live status, color-coded agent badges, and a model picker for switching providers on the fly.

### The Agents

| Agent             | Role                                                                                                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **supervisor**    | Orchestrator. Inspects the conversation and routes to the next specialist or finishes. Uses structured output (`SupervisorDecision`) to commit to a routing choice with reasoning. |
| **specification** | Turns vague requests into actionable specs with scope, requirements, and acceptance criteria. Routed to first when a task is ambiguous.                                            |
| **researcher**    | Investigates topics, synthesizes knowledge from multiple angles, and produces structured factual summaries.                                                                        |
| **coder**         | Writes production-quality code and refactors for clarity and performance.                                                                                                          |
| **analyst**       | Performs quantitative reasoning, statistical analysis, and data-driven evaluation.                                                                                                 |
| **tester**        | Designs test strategies, identifies edge cases, and writes concrete test code.                                                                                                     |
| **debugger**      | Systematically diagnoses bugs and tracebacks, then recommends root-cause fixes.                                                                                                    |

### Features

- **Supervised routing** — a dedicated supervisor node makes every routing decision, so workflows stay coherent across many turns.
- **Multi-provider model support** — swap between Anthropic (Claude Opus/Sonnet/Haiku 4.x) and OpenAI (GPT-5.x, GPT-4.x) models at runtime via a modal picker.
- **Streaming TUI** — every routing decision, specialist response, and tool call is rendered live with agent-specific colors and icons.
- **Persistent conversation history** — context is carried across prompts within a single session so agents can iterate on prior work.
- **File-saving tool** — any worker can call `save_file` to write content directly to disk.

## Installation

### Prerequisites

- Python **3.10+**
- An API key for at least one supported provider (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`)

### Setup

```bash
git clone https://github.com/RishabSA/AgentForge.git
cd AgentForge

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root with whichever keys you have:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Only providers with a valid key will be selectable in the model picker; the rest are grayed out.

## Usage

Launch the AgentForge Terminal User Interface (TUI) with:

```bash
python main.py
```

Enter a prompt at the bottom of the screen and press **Enter**. The supervisor will inspect your request, route it to the appropriate specialist(s), and stream each step of the conversation into the log.

### Keybindings

| Key              | Action                     |
| ---------------- | -------------------------- |
| `Enter`          | Submit prompt              |
| `Ctrl+K`         | Open the model picker      |
| `Ctrl+L`         | Clear the conversation log |
| `Ctrl+C` / `Esc` | Quit                       |

### Example prompts

- _"Build me a small CLI tool in Python that renames image files based on their EXIF date, and save it to `rename_images.py`."_
- _"Here's a traceback from my FastAPI app — find the root cause and propose a fix."_
- _"Compare the runtime cost of three caching strategies for a 10M-row read-heavy workload."_
- _"Write unit tests for the `parse_config` function I just described."_

### Configuration

Defaults live in `agents.py`:

```python
MODEL_ID = "anthropic:claude-sonnet-4-6"
TEMPERATURE = 1.0
WORKERS = ["researcher", "coder", "analyst", "specification", "tester", "debugger"]
```

Edit these to change the startup model, sampling temperature, or the set of workers registered in the graph. To add a new model to the picker, append an entry to `MODEL_CATALOG` in `agents.py`.

## Architecture

```
                       ┌──────────────┐
           user ──────▶│  supervisor  │──── FINISH ────▶ END
                       └──────┬───────┘
                              │
          ┌──────────┬────────┼────────┬──────────┬──────────┐
          ▼          ▼        ▼        ▼          ▼          ▼
      researcher  coder   analyst   spec      tester     debugger
          │          │        │        │          │          │
          └──────────┴────┬───┴────────┴──────────┴──────────┘
                          │
                (tool_calls present?)
                          │
                      ┌───┴───┐
                      ▼       ▼
                    save    END
                      │
                      ▼
                     END
```

- **State** (`AgentState`) — a `TypedDict` carrying the message history, the next routing target, and a completion flag. Messages accumulate via LangGraph's `add_messages` reducer.
- **Supervisor node** — calls the model with `with_structured_output(SupervisorDecision)` to force a valid routing choice plus reasoning, then emits a routing message and sets `next_agent`.
- **Worker nodes** — each calls the same model bound to the `save_file` tool with a specialist system prompt from `prompts.py`.
- **Tool node (`save`)** — a LangGraph `ToolNode` that executes any `save_file` calls a worker emitted, then returns control to the end of the turn.
- **Streaming** — `run()` is a generator that wraps `app.stream(...)` and yields `AgentEvent` dicts (`node`, `content`, `event_type`, `message`) which the TUI consumes from a background worker thread.

## File Structure

```
AgentForge/
├── main.py            # Textual TUI: status bar, model picker, streaming log
├── agents.py          # LangGraph StateGraph, supervisor/worker nodes, tools
├── prompts.py         # System prompts for supervisor and all six specialists
├── requirements.txt   # langgraph, langchain, textual, rich
├── .env               # API keys (not committed)
├── LICENSE
└── README.md
```
