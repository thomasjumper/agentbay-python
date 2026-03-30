# AgentBay Python SDK

Persistent memory for AI agents. 3 lines to give your agent a brain.

## Install

```bash
pip install agentbay
```

## Quick Start

```python
from agentbay import AgentBay

brain = AgentBay("ab_live_your_key", project_id="your-project-id")
brain.store("Next.js 16 + Prisma + PostgreSQL", title="Project stack")
results = brain.recall("What stack does this project use?")
```

Or create a new brain on the fly:

```python
from agentbay import AgentBay

brain = AgentBay("ab_live_your_key")
brain.setup_brain("My Agent's Memory")
brain.store("Always use UTC timestamps", title="Convention", type="PREFERENCE")
```

## Core API

| Method | What it does |
|--------|-------------|
| `brain.store(content, title, type, tier, tags)` | Save a memory |
| `brain.recall(query, limit, tier, tags)` | Search memories (semantic + keyword) |
| `brain.forget(knowledge_id)` | Archive a memory |
| `brain.verify(knowledge_id)` | Confirm a memory is still accurate |
| `brain.health()` | Get memory stats |
| `brain.setup_brain(name, description)` | Create a new Knowledge Brain |

## Memory Types

- `PATTERN` -- Learned behaviors and recurring themes
- `FACT` -- Verified information
- `PREFERENCE` -- User/agent preferences
- `PROCEDURE` -- Step-by-step processes
- `CONTEXT` -- Situational context

## With CrewAI

```bash
pip install agentbay[crewai]
```

```python
from crewai import Agent
from agentbay.integrations.crewai import AgentBayCrewAIMemory

memory = AgentBayCrewAIMemory(
    api_key="ab_live_your_key",
    project_id="your-project-id",
)

agent = Agent(
    role="Researcher",
    goal="Find and remember information",
    memory=memory,
)
```

## With LangChain

```bash
pip install agentbay[langchain]
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agentbay.integrations.langchain import AgentBayMemoryTool

tool = AgentBayMemoryTool(
    api_key="ab_live_your_key",
    project_id="your-project-id",
)

llm = ChatOpenAI()
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
)
agent.run("Remember that deploys happen every Tuesday at 2pm UTC")
```

## Error Handling

```python
from agentbay.client import AgentBayError, AuthenticationError, RateLimitError

try:
    results = brain.recall("query")
except AuthenticationError:
    print("Bad API key")
except RateLimitError:
    print("Slow down")
except AgentBayError as e:
    print(f"Error {e.status_code}: {e}")
```

## Links

- [AgentBay](https://www.aiagentsbay.com) -- AI agent marketplace + memory platform
- [API Docs](https://www.aiagentsbay.com/docs)
- [MCP Server](https://www.npmjs.com/package/agentbay-mcp)
