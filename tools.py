import os
from dotenv import load_dotenv

load_dotenv()


def search(query: str) -> str:
    """Call the Tavily Search API and return a clean result string.

    Returns an error string on failure — never raises an exception.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        result = client.search(query=query, max_results=3)
        snippets = [r.get("content", "").strip() for r in result.get("results", []) if r.get("content")]
        return "\n---\n".join(snippets) if snippets else "No results found."
    except KeyError:
        return "Search error: TAVILY_API_KEY not set in environment."
    except Exception as e:
        return f"Search error: {e}"


# Registry mapping action names to callable tools.
TOOLS: dict[str, callable] = {
    "Search": search,
}
