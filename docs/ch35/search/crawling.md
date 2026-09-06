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

## Exercises

**Exercise 1.**
Model a web crawler as a graph traversal algorithm. Compare BFS and DFS crawling strategies in terms of page quality and resource usage.

??? success "Solution to Exercise 1"
    The web is a directed graph: pages are vertices, hyperlinks are edges. **BFS**: uses a FIFO queue of URLs. Discovers pages in order of link distance from seed pages. Tends to visit high-quality pages first (popular pages are linked early). Requires storing the entire frontier in memory (can be very large). **DFS**: uses a LIFO stack. Follows link chains deep before backtracking. May descend into low-quality or infinite-depth subgraphs (e.g., calendar pages with infinitely many dates). Uses less memory (stack depth is bounded by the maximum path length). In practice, crawlers use **priority-based BFS**: the frontier is a priority queue ordered by estimated page importance (e.g., PageRank estimate, domain authority). This combines BFS's breadth with quality-aware ordering. $\square$

---

**Exercise 2.**
A web crawler must respect robots.txt politeness rules, limiting requests to 1 per second per domain. Design a scheduling algorithm that maximizes throughput while respecting this constraint.

??? success "Solution to Exercise 2"
    Maintain a per-domain queue and a global priority queue of domains. Each domain queue holds pending URLs. The global queue orders domains by their "next allowed request time" (1 second after the last request to that domain). The crawler loop: (1) dequeue the domain with the earliest allowed time. (2) If the current time $\ge$ allowed time, fetch the next URL from that domain's queue. (3) Update the domain's allowed time to now + 1 second. (4) Re-insert the domain into the global queue (or remove it if its URL queue is empty). With $D$ active domains, this achieves throughput of $D$ pages/second (1 page/sec/domain $\times D$ domains in parallel). For $D = 10{,}000$ domains, throughput is 10,000 pages/second. The priority queue operations are $O(\log D)$ per fetch. The key insight: parallelism across domains compensates for the per-domain rate limit. $\square$

---

**Exercise 3.**
Explain how a Bloom filter is used in web crawling to avoid revisiting URLs. What happens when the Bloom filter becomes too full?

??? success "Solution to Exercise 3"
    Before adding a discovered URL to the frontier, the crawler checks a Bloom filter. If the filter says "possibly present," the URL is skipped (already crawled or already in the frontier). If "definitely absent," the URL is added and the filter is updated. With $10^9$ URLs and a 1% false positive rate, the Bloom filter uses $\sim$1.2 GB -- far less than storing $10^9$ full URLs ($\sim$100 GB). When the filter becomes too full (false positive rate exceeds threshold): (1) the crawler may stop discovering new URLs (false positives reject legitimate new URLs). Mitigation: build a new, larger Bloom filter and repopulate it from the canonical URL database. (2) Alternatively, use a scalable Bloom filter that adds new filter segments when the capacity is exceeded, or use a partitioned Bloom filter with independent filters per domain. $\square$

---

**Exercise 4.**
A focused crawler aims to collect pages about "machine learning" rather than crawling the entire web. Describe how link priority and content relevance scoring guide the crawl.

??? success "Solution to Exercise 4"
    A focused crawler maintains a **relevance classifier** trained on seed pages. For each discovered page: (1) download and extract text. (2) Compute a relevance score (e.g., TF-IDF similarity to the "machine learning" topic model, or a trained classifier's confidence). (3) Extract outgoing links. For each link, estimate its relevance using: the parent page's relevance, the anchor text (if it contains "deep learning," "neural network," etc.), and the link's domain reputation. (4) Add links to the priority queue ordered by estimated relevance. The crawler preferentially follows links from relevant pages with relevant anchor text. This keeps the crawl focused: after a few hops from seed pages, most discovered pages are on-topic. The tradeoff: the crawler may miss relevant pages reachable only through irrelevant intermediaries (the "tunnel" problem). $\square$

---

**Exercise 5.**
Estimate the storage and bandwidth requirements for crawling 10 billion web pages with an average page size of 50 KB, refreshing the entire crawl every 30 days.

??? success "Solution to Exercise 5"
    **Storage**: $10^{10} \times 50 \text{ KB} = 5 \times 10^{11} \text{ KB} = 500$ TB raw. With compression (HTML compresses $\sim$5:1): $\sim$100 TB. Plus metadata (URLs, timestamps, HTTP headers): $\sim$10 TB. Total: $\sim$110 TB. **Bandwidth**: $500 \text{ TB} / 30 \text{ days} = 16.7 \text{ TB/day} = 193 \text{ GB/hour} = 430 \text{ Mbps}$ sustained. This is feasible with a cluster of machines, each with a 1 Gbps connection. **Pages per second**: $10^{10} / (30 \times 86400) \approx 3{,}858$ pages/sec. With 1-second politeness delays per domain and $\sim$100 million active domains, only a small fraction of domains are being accessed at any moment. The bottleneck is typically DNS resolution, network latency, and storage write throughput, not raw bandwidth. $\square$
