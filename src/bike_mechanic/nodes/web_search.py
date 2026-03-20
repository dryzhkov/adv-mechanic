"""Web search node for spec validation."""

from bike_mechanic.config import TAVILY_API_KEY
from bike_mechanic.state import GraphState, WebResult

SEARCH_DOMAINS = [
    "advrider.com",
    "reddit.com",
    "thumpertalk.com",
    "ktmforums.com",
    "husqvarnaownersgroup.com",
]


def web_search(state: GraphState) -> dict:
    """Search the web to validate or supplement manual data."""
    search_query = state.get("web_search_query", state["query"])
    bike_model = state.get("bike_model", "")

    if not TAVILY_API_KEY:
        print("  [web_search] TAVILY_API_KEY not set, skipping web search")
        return {"web_results": []}

    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)

    queries_with_domains: list[tuple[str, list[str]]] = [
        (search_query, SEARCH_DOMAINS),
    ]
    if bike_model:
        queries_with_domains.append((search_query, ["advrider.com"]))
        queries_with_domains.append((search_query, ["reddit.com"]))

    results: list[WebResult] = []
    seen_urls: set[str] = set()

    for q, domains in queries_with_domains[:3]:
        try:
            response = client.search(
                query=q,
                search_depth="advanced",
                max_results=3,
                include_domains=domains,
            )
            for r in response.get("results", []):
                url = r["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    results.append(
                        WebResult(
                            text=r.get("content", ""),
                            source=r.get("title", ""),
                            url=url,
                        )
                    )
        except Exception as e:
            print(f"  [web_search] Error searching '{q}': {e}")

    return {"web_results": results}
