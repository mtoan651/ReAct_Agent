from agent import ReActAgent

TASKS = [
    # Task 1: Planning & Quantitative Reasoning
    "What fraction of Japan's population is Taiwan's population as of 2025?",
    # Task 2: Technical Specificity
    "Compare the main display specs of iPhone 15 and Samsung S24.",
    # Task 3: Resilience & Reflection Test
    "Who is the CEO of the startup 'Morphic' AI search?",
]


def main():
    # A single ReActAgent instance is reused for all tasks (assignment constraint).
    agent = ReActAgent()

    for i, task in enumerate(TASKS, start=1):
        print(f"\n{'=' * 60}")
        print(f"Task {i}: {task}")
        print("=" * 60)
        answer = agent.execute(task)
        print(f"\n>>> Final Answer: {answer}")


if __name__ == "__main__":
    main()
