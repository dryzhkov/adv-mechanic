"""MCP server exposing web search tools for motorcycle forums."""

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

from bike_mechanic.config import MOTO_DOMAINS, TAVILY_API_KEY

mcp = FastMCP("bike-mechanic-web")


def _search(query: str, domains: list[str] | None = None, max_results: int = 5) -> str:
    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY not set. Configure it in .env to enable web search."

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            **({"include_domains": domains} if domains else {}),
        )
    except Exception as e:
        return f"Search error: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    parts = []
    for r in results:
        parts.append(
            f"[{r.get('title', 'No title')}]\n"
            f"URL: {r['url']}\n"
            f"{r.get('content', '')}\n"
        )

    return "\n---\n".join(parts)


@mcp.tool()
def search_advrider(query: str, bike_model: str = "") -> str:
    """Search ADVRider forums for motorcycle information.

    Args:
        query: Search query
        bike_model: Optional bike model to include in search
    """
    full_query = f"{bike_model} {query}".strip() if bike_model else query
    return _search(full_query, domains=["advrider.com"])


@mcp.tool()
def search_reddit(query: str, subreddit: str = "") -> str:
    """Search Reddit for motorcycle discussions.

    Args:
        query: Search query
        subreddit: Optional subreddit (e.g., "KTM", "motorcycles")
    """
    if subreddit:
        query = f"r/{subreddit} {query}"
    return _search(query, domains=["reddit.com"])


@mcp.tool()
def search_thumpertalk(query: str, bike_model: str = "") -> str:
    """Search ThumperTalk forums for motorcycle information.

    Args:
        query: Search query
        bike_model: Optional bike model to include in search
    """
    full_query = f"{bike_model} {query}".strip() if bike_model else query
    return _search(full_query, domains=["thumpertalk.com"])


@mcp.tool()
def search_general(query: str) -> str:
    """General web search across all motorcycle forums and sites.

    Args:
        query: Search query
    """
    return _search(query, domains=MOTO_DOMAINS)


if __name__ == "__main__":
    mcp.run()
