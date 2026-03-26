# Web Crawling

A search engine begins with a **web crawler** -- a program that systematically discovers and downloads web pages by following hyperlinks. From an algorithmic perspective, the web is a directed graph where pages are vertices and hyperlinks are edges. Crawling is graph traversal at massive scale, with billions of vertices and constraints imposed by network latency, politeness policies, and storage limits.

## The Web Graph

The web can be modeled as a directed graph $G = (V, E)$:

- Each URL is a vertex $v \in V$.
- A hyperlink from page $u$ to page $v$ is a directed edge $(u, v) \in E$.

The web graph has specific structural properties:

- **Power-law degree distribution**: A few pages have millions of inlinks; most pages have very few.
- **Bow-tie structure**: A large strongly connected component (SCC) at the core, with pages that link in but not out (IN), pages linked to but not linking back (OUT), and disconnected tendrils.

## Crawl Strategy

### BFS Crawling

Breadth-first search from a set of **seed URLs** discovers pages level by level:

1. Initialize a queue (frontier) with seed URLs.
2. Dequeue a URL, download the page, extract links.
3. Enqueue newly discovered URLs (not already seen).

BFS tends to discover high-quality pages first (pages close to well-known seeds).

### Priority-Based Crawling

Replace the FIFO queue with a priority queue. Assign priority based on:

- **PageRank estimate**: Crawl important pages first.
- **Freshness**: Re-crawl pages that change frequently.
- **Domain diversity**: Avoid over-crawling a single domain.

## URL Frontier

The URL frontier manages the queue of URLs to visit. At web scale, it must:

- **Dedup**: Use a Bloom filter or hash set to avoid revisiting URLs. With $n$ URLs and a Bloom filter of $m$ bits:

$$
P_{\text{false positive}} \approx \left(1 - e^{-kn/m}\right)^k
$$

- **Politeness**: Maintain per-domain queues with rate limiting (e.g., one request per second per domain).
- **Persistence**: Store the frontier on disk since it may contain billions of URLs.

## Politeness and robots.txt

Crawlers must respect `robots.txt` directives and rate-limit requests to each domain. A common policy:

$$
\text{delay between requests to domain } d = \max(\text{robots\_delay}(d),\; \Delta_{\min})
$$

where $\Delta_{\min}$ is the crawler's minimum politeness interval (typically 1--10 seconds).

## Implementation

```python
"""
Web Crawler -- BFS-based crawl simulation with URL deduplication.

Simulates crawling a directed web graph starting from seed URLs,
using BFS traversal and a visited set for deduplication.
"""

from __future__ import annotations
from collections import deque


# === Simulated Web ============================================================

def build_web() -> dict[str, list[str]]:
    """Create a small simulated web graph."""
    return {
        "seed.com": ["a.com", "b.com", "c.com"],
        "a.com": ["d.com", "e.com"],
        "b.com": ["a.com", "f.com"],
        "c.com": ["f.com"],
        "d.com": ["seed.com"],
        "e.com": ["g.com"],
        "f.com": ["g.com", "a.com"],
        "g.com": [],
    }


# === BFS Crawler ==============================================================

def bfs_crawl(web: dict[str, list[str]],
              seeds: list[str],
              max_pages: int = 100) -> list[str]:
    """Crawl using BFS from *seeds*. Returns pages in discovery order."""
    visited: set[str] = set()
    frontier: deque[str] = deque()
    crawl_order: list[str] = []

    for seed in seeds:
        if seed not in visited:
            frontier.append(seed)
            visited.add(seed)

    while frontier and len(crawl_order) < max_pages:
        url = frontier.popleft()
        crawl_order.append(url)

        # Extract links from the page
        links = web.get(url, [])
        for link in links:
            if link not in visited:
                visited.add(link)
                frontier.append(link)

    return crawl_order


# === DFS Crawler ==============================================================

def dfs_crawl(web: dict[str, list[str]],
              seeds: list[str],
              max_pages: int = 100) -> list[str]:
    """Crawl using DFS from *seeds*. Returns pages in discovery order."""
    visited: set[str] = set()
    stack: list[str] = list(reversed(seeds))
    crawl_order: list[str] = []

    for seed in seeds:
        visited.add(seed)

    while stack and len(crawl_order) < max_pages:
        url = stack.pop()
        crawl_order.append(url)

        links = web.get(url, [])
        for link in reversed(links):
            if link not in visited:
                visited.add(link)
                stack.append(link)

    return crawl_order


# === Main =====================================================================

if __name__ == "__main__":
    web = build_web()
    seeds = ["seed.com"]

    bfs_result = bfs_crawl(web, seeds)
    dfs_result = dfs_crawl(web, seeds)

    print(f"Web graph: {len(web)} pages")
    print(f"\nBFS crawl order ({len(bfs_result)} pages):")
    for i, url in enumerate(bfs_result):
        print(f"  {i+1}. {url}")

    print(f"\nDFS crawl order ({len(dfs_result)} pages):")
    for i, url in enumerate(dfs_result):
        print(f"  {i+1}. {url}")
```

**Output:**

```
Web graph: 8 pages

BFS crawl order (8 pages):
  1. seed.com
  2. a.com
  3. b.com
  4. c.com
  5. d.com
  6. e.com
  7. f.com
  8. g.com

DFS crawl order (8 pages):
  1. seed.com
  2. a.com
  3. d.com
  4. e.com
  5. g.com
  6. b.com
  7. f.com
  8. c.com
```

BFS discovers pages level by level (all direct links from seed first), which tends to find important pages early. DFS dives deep along one path before backtracking, which can discover deep content sooner but may miss breadth.

## Reference

- Olston, C. and Najork, M. "Web Crawling." *Foundations and Trends in Information Retrieval*, 2010
- Manning, C.D., Raghavan, P., and Schutze, H. *Introduction to Information Retrieval*. Cambridge University Press
