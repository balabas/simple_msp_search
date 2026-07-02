from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from ddgs import DDGS
import trafilatura
import httpx
from markdownify import markdownify as md
import asyncio
import datetime
import yaml
from pathlib import Path


class LiteralDumper(yaml.Dumper):
    pass

def _literal_str(dumper, data):
    if "\n" in data:
        cleaned = "\n".join(line.rstrip() for line in data.split("\n"))
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

LiteralDumper.add_representer(str, _literal_str)

mcp = FastMCP("DuckDuckGoSearch")

LOCAL_LLM_BASE = "http://127.0.0.1:8080"
default_page_max_char_length = 16000
LOG_FILE = Path(__file__).parent / "debug_log/log.yaml"
LOG_MAX_BYTES = 2 * 1024 * 1024  # 2 MB


def log_call(tool: str, params: dict, mcp_response: str, fetched_pages: list = None, llm_responses: list = None, error: str = None):
    entry = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tool": tool, "params": params}
    if fetched_pages:
        entry["fetched_pages"] = fetched_pages
    if llm_responses:
        entry["llm_responses"] = llm_responses
    entry["mcp_response"] = mcp_response
    if error:
        entry["error"] = error

    existing = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            existing = yaml.safe_load(f) or []

    existing.append(entry)

    # Trim oldest entries until serialized size fits within limit
    while len(existing) > 1:
        if len(yaml.dump(existing, Dumper=LiteralDumper, allow_unicode=True, sort_keys=False).encode()) <= LOG_MAX_BYTES:
            break
        existing.pop(0)

    with open(LOG_FILE, "w") as f:
        yaml.dump(existing, f, Dumper=LiteralDumper, allow_unicode=True, sort_keys=False)


def get_local_llm_model() -> str:
    response = httpx.get(f"{LOCAL_LLM_BASE}/v1/models", timeout=5.0)
    return response.json()["data"][0]["id"]


async def filter_with_local_llm(client: httpx.AsyncClient, content: str, query: str) -> tuple[str, dict]:
    """Returns (filtered_content, raw_llm_json_response)."""
    prompt = (
        f'''You are a content filter. The user searched for: >>>
{query}
<<<

Below is extracted text from a web page. Extract and return ONLY the parts
that are directly relevant to the search query. Remove anything unrelated, such as ads, navigation menus, page footers, page headers and other irrelevant content.
If nothing is relevant, respond with: [not relevant]
PAGE CONTENT:>>>
{content}
<<<'''
    )
    response = await client.post(
        f"{LOCAL_LLM_BASE}/v1/chat/completions",
        json={
            "model": get_local_llm_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024 * 8,
        },
        timeout=60.0 * 5,
    )
    raw = response.json()
    filtered = raw["choices"][0]["message"]["content"].strip()
    return filtered, raw


async def fetch_and_convert_async(
    client: httpx.AsyncClient, title: str, url: str, query: str = ""
) -> tuple[str, str, dict | None]:
    """Returns (mcp_result, raw_page_md, llm_raw_json_or_None)."""
    try:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        content_md = md(response.text)
        raw_page_md = "\n".join(
            [line for line in content_md.splitlines() if line.strip()][:default_page_max_char_length]
        )
        llm_raw = None
        filtered_md = raw_page_md
        if query:
            filtered_md, llm_raw = await filter_with_local_llm(client, raw_page_md, query)
        result = f"## {title}\nURL: {url}\n\n{filtered_md}\n\n---"
        return result, raw_page_md, llm_raw
    except Exception as e:
        result = f"## {title}\nURL: {url}\n*Ошибка при загрузке: {str(e)}*\n\n---"
        return result, "", None


async def fetch_all_pages(results, query: str = "") -> tuple[list[str], list[dict], list[dict]]:
    """Returns (mcp_pages, fetched_pages_log, llm_responses_log)."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_and_convert_async(client, res["title"], res["href"], query)
            for res in results
        ]
        triples = await asyncio.gather(*tasks)

    mcp_pages = [r for r, _, _ in triples]
    fetched_pages = [{"url": res["href"], "raw_md": raw} for res, (_, raw, _) in zip(results, triples)]
    llm_responses = [
        {"url": res["href"], "llm_response": llm}
        for res, (_, _, llm) in zip(results, triples)
        if llm is not None
    ]
    return mcp_pages, fetched_pages, llm_responses


@mcp.tool()
def get_current_date_time() -> str:
    """Returns the current date and time in the format YYYY-MM-DD HH:MM:SS."""
    result = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_call("get_current_date_time", {}, mcp_response=result)
    return result


@mcp.tool()
def fetch_and_convert(url: str = "", title: str = "", query: str = "") -> str:
    """
    Fetch URL content and convert to markdown.
    Optionally pass query to filter content with the local LLM before returning.
    """
    async def _run():
        async with httpx.AsyncClient() as client:
            return await fetch_and_convert_async(client, title, url, query)

    result, raw_page_md, llm_raw = asyncio.run(_run())
    log_call(
        "fetch_and_convert",
        {"url": url, "title": title, "query": query},
        mcp_response=result,
        fetched_pages=[{"url": url, "raw_md": raw_page_md}],
        llm_responses=[{"url": url, "llm_response": llm_raw}] if llm_raw else None,
    )
    return result


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the Internet using DuckDuckGo.
    Fetches each result page and filters content through a local LLM before returning.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if not results:
                log_call("search_web", {"query": query, "max_results": max_results}, mcp_response="No results found.")
                return "No results found."
    except Exception as e:
        log_call("search_web", {"query": query, "max_results": max_results}, mcp_response="", error=str(e))
        return f"Error occurred while searching: {str(e)}"

    mcp_pages, fetched_pages, llm_responses = asyncio.run(fetch_all_pages(results, query=query))
    result = "\n\n".join(mcp_pages)
    log_call(
        "search_web",
        {"query": query, "max_results": max_results},
        mcp_response=result,
        fetched_pages=fetched_pages,
        llm_responses=llm_responses or None,
    )
    return result


if __name__ == "__main__":
    cors_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8325,
        path="/mcp",
        middleware=cors_middleware,
    )
