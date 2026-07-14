# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run a Jupyter notebook kernel (notebooks run interactively in Cursor/VS Code)
uv run jupyter notebook

# Run main.py
uv run python main.py

# Add a dependency
uv add <package>
```

## Environment Setup

Copy `.env` with the required keys before running any notebooks:
- `GROQ_API_KEY` — Groq LLM API access
- `TAVILY_API_KEY` — Tavily web search tool

Notebooks load these via `python-dotenv` (`load_dotenv()`).

## Architecture

This is a self-study repo exploring LangGraph concepts, organized as numbered Jupyter notebooks. Each notebook is standalone and demonstrates a specific concept:

### `1-BasicChatBot/`

| Notebook | Concept |
|---|---|
| `1-basicchatBot.ipynb` | Minimal `StateGraph` with a single LLM node; `START → chatbot → END` |
| `1.1-basicchatBotwitTools.ipynb` | Tool-augmented graph using `ToolNode` + `tools_condition` for conditional routing; parallel tool calls; `MemorySaver` for conversation memory across `thread_id` config; streaming with `.stream()` (`"updates"` vs `"values"` modes) and async `astream_events` |
| `1.2-humanfeedbackintheLoop.ipynb` | Human-in-the-loop via `langgraph.types.interrupt` and `Command(resume=...)` to pause graph execution and inject human responses |

### Core LangGraph patterns used

- **State**: `TypedDict` with `messages: Annotated[list, add_messages]` — messages are appended, not replaced
- **Nodes**: Python functions returning partial state dicts; added via `builder.add_node(name, fn)`
- **Edges**: `add_edge` for fixed transitions; `add_conditional_edges` with `tools_condition` for LLM-decides-to-call-tool branching
- **Memory**: `MemorySaver` checkpointer compiled into the graph; conversations scoped by `{"configurable": {"thread_id": "<id>"}}`
- **Tools**: Bound to LLM via `llm.bind_tools(tools)`; executed by `ToolNode`; custom tools defined as plain functions with docstrings or via `@tool` decorator

### LLM / Model

All notebooks use `ChatGroq` with model `qwen/qwen3-32b` (a reasoning model — responses include `<think>` blocks). Initialized either directly:
```python
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="qwen/qwen3-32b")
```
or via LangChain's generic init:
```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:qwen/qwen3-32b")
```
