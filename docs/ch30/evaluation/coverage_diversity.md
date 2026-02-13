# Coverage and Diversity

## Overview

Beyond accuracy, recommender systems should provide diverse recommendations that cover a broad range of items. A system that only recommends popular items may be accurate but unhelpful.

## Coverage

**Catalog coverage**: fraction of items ever recommended:

$$\text{Coverage} = \frac{|\text{unique items recommended}|}{|\text{all items}|}$$

Low coverage indicates popularity bias — the system ignores the long tail.

## Diversity

**Intra-list diversity**: average pairwise dissimilarity within a recommendation list:

$$\text{Diversity}(L) = \frac{2}{|L|(|L|-1)} \sum_{i < j} (1 - \text{sim}(i, j))$$

where $\text{sim}$ can be content-based similarity, co-occurrence, or embedding distance.

## Novelty

**Expected popularity complement**: how surprising are the recommendations?

$$\text{Novelty}(L) = \frac{1}{|L|} \sum_{i \in L} -\log_2 p(i)$$

where $p(i)$ is the popularity of item $i$. Recommending rare items scores higher.

## Serendipity

Measures unexpected but relevant recommendations — items the user would not have found on their own but enjoys:

$$\text{Serendipity} = \frac{|\text{relevant} \cap \text{unexpected}|}{|\text{recommendations}|}$$

## Accuracy-Diversity Tradeoff

Optimizing for accuracy alone typically reduces diversity (popular items are safe bets). Methods to improve diversity without sacrificing too much accuracy include re-ranking with diversity constraints (MMR), DPP-based diversification, and exploration/exploitation balancing.
