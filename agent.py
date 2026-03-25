import os
import re
from dotenv import load_dotenv
from tools import TOOLS

load_dotenv()

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a ReAct agent that answers questions by alternating between Thinking, Acting, and Observing.

Always follow this EXACT format:

Thought: <your reasoning about what to do next>
Action: <tool name — must be one of: Search>
Action Input: <query or input for the tool>
Observation: <result returned by the tool — provided by the system, never write this yourself>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: <your final reasoning once you have enough information>
Final Answer: <your complete, direct answer to the original question>

Rules:
- You MUST begin every response with "Thought:".
- NEVER write an "Observation:" line yourself — the system injects it after each Action.
- If a search returns empty or irrelevant results, reflect on why and try a differently phrased query.
- For multi-part questions, decompose into sequential searches before computing the answer.
- You have at most 5 steps total. Be concise and efficient.

=== EXAMPLE ===
Question: What is the population of Australia?

Thought: The user wants Australia's current population. I will search for the latest figure.
Action: Search
Action Input: Australia population 2025
Observation: Australia's population is approximately 26.8 million as of 2025, according to the Australian Bureau of Statistics.

Thought: I have the answer I need.
Final Answer: Australia's population is approximately 26.8 million as of 2025.
=== END EXAMPLE ===

Now answer the following question.
"""


class ReActAgent:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system = system_prompt
        self.messages: list[str] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Send the accumulated prompt to Groq and return the raw text.

        The stop sequence '\nObservation:' prevents the model from
        hallucinating its own observation after writing an Action.
        """
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": "".join(self.messages)},
            ],
            stop=["\nObservation:"],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _parse_action(self, llm_response: str) -> tuple[str | None, str | None]:
        """Extract (action_name, action_input) from the LLM response.

        Returns (None, None) when no Action line is found.
        """
        action_match = re.search(r"^Action:\s*(.+)$", llm_response, re.MULTILINE)
        input_match = re.search(r"^Action Input:\s*(.+)$", llm_response, re.MULTILINE)
        if action_match and input_match:
            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip().strip('"').strip("'")
            return action_name, action_input
        return None, None

    def _parse_final_answer(self, llm_response: str) -> str | None:
        """Return the Final Answer text if present, else None."""
        match = re.search(r"^Final Answer:\s*(.+)", llm_response, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _build_prompt(self) -> str:
        """Return the full conversation history (system prompt is sent separately)."""
        return "".join(self.messages)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, query: str) -> str:
        """Run the ReAct loop for *query* and return the final answer string."""
        # Reset history for each new query (same instance, fresh context).
        self.messages = [f"Question: {query}\n\n"]

        for step in range(1, MAX_ITERATIONS + 1):
            print(f"\n[Step {step}]")

            # 1. Call LLM with current context.
            prompt = self._build_prompt()
            llm_response = self._call_llm(prompt)
            print(llm_response)

            # 2. Append LLM output to history.
            self.messages.append(llm_response)

            # 3. Check for final answer — we're done.
            final_answer = self._parse_final_answer(llm_response)
            if final_answer:
                return final_answer

            # 4. Parse and dispatch a tool action.
            action_name, action_input = self._parse_action(llm_response)
            if action_name and action_name in TOOLS:
                observation = TOOLS[action_name](action_input)
            elif action_name:
                observation = f"Unknown tool '{action_name}'. Available tools: {list(TOOLS.keys())}."
            else:
                # LLM produced neither an action nor a final answer.
                observation = "No action detected. Please specify an Action or provide a Final Answer."

            print(f"Observation: {observation}")

            # 5. Inject the observation back into the context for the next step.
            self.messages.append(f"\nObservation: {observation}\n")

        return "Max iterations reached without a final answer."
