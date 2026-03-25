# ReAct Agent

A ReAct (Reasoning + Acting) agent implementation that answers complex queries through an iterative Thought → Action → Observation loop. Built with Groq (llama-3.3-70b-versatile) as the LLM and Tavily as the search tool.

## How It Works

The agent follows a structured cognitive loop for each query:

```
Question
   └─> Thought  (reasoning about what to do)
   └─> Action   (call a tool)
   └─> Observation (tool result, injected by system)
   └─> Thought  (reflect, decide next step)
   └─> ...
   └─> Final Answer
```

Key design decisions:
- **Few-shot prompting**: a complete worked example is embedded in the system prompt so the model knows the exact format expected
- **Stop sequence** (`\nObservation:`): prevents the LLM from hallucinating its own observations — the system always injects real tool output
- **Iteration cap** (5 steps max): avoids infinite loops
- **Single agent instance**: the same `ReActAgent` object is reused across all tasks; `execute()` resets message history per query

## Project Structure

```
ReAct_Agent/
├── agent.py          # ReActAgent class — the core loop
├── tools.py          # Search tool wrapper (Tavily API)
├── main.py           # Entry point — defines and runs 3 benchmark tasks
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── .gitignore
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ReAct_Agent
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- **Groq API key**: [console.groq.com](https://console.groq.com)
- **Tavily API key**: [app.tavily.com](https://app.tavily.com)

### 3. Run

```bash
python main.py
```

## Benchmark Tasks

The agent is evaluated on three tasks that test different capabilities:

| Task | Query | Capability Tested |
|------|-------|-------------------|
| 1 | What fraction of Japan's population is Taiwan's population as of 2025? | Multi-step planning & quantitative reasoning |
| 2 | Compare the main display specs of iPhone 15 and Samsung S24. | Technical specificity & accurate retrieval |
| 3 | Who is the CEO of the startup 'Morphic' AI search? | Resilience & self-correction on ambiguous queries |

## Example Output

```
============================================================
Task 1: What fraction of Japan's population is Taiwan's population as of 2025?
============================================================

[Step 1]
Thought: I need both populations. Let me search for Japan's first.
Action: Search
Action Input: Japan population 2025
Observation: Japan's population is approximately 123.8 million as of 2025...

[Step 2]
Thought: Now I need Taiwan's population.
Action: Search
Action Input: Taiwan population 2025
Observation: Taiwan's population is approximately 23.4 million as of 2025...

[Step 3]
Thought: 23.4 / 123.8 ≈ 0.189, roughly 1/5.
Final Answer: Taiwan's population is approximately 1/5 (about 18.9%) of Japan's population as of 2025.

>>> Final Answer: Taiwan's population is approximately 1/5 (about 18.9%) of Japan's population as of 2025.
```

## Architecture Details

### `agent.py` — `ReActAgent`

| Method | Purpose |
|--------|---------|
| `__init__(system_prompt)` | Initializes agent with few-shot system prompt |
| `execute(query)` | Runs the ReAct loop; resets history per call |
| `_call_llm(prompt)` | Sends prompt to Groq with stop sequence |
| `_parse_action(response)` | Extracts `Action` and `Action Input` via regex |
| `_parse_final_answer(response)` | Extracts `Final Answer` via regex |

### `tools.py`

Wraps the Tavily Search API. Returns top-3 result snippets joined by `---`. All errors are returned as strings — the tool never raises an exception, so the agent can reflect and retry.

```python
TOOLS = {
    "Search": search,   # add more tools here
}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | LLM inference (llama-3.3-70b-versatile) |
| `tavily-python` | Web search API |
| `python-dotenv` | Load `.env` into environment |
| `google-genai` | Available for future use |
