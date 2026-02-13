# Conditional Independence

## Definition

$X \perp Y \mid Z$ if $P(X, Y | Z) = P(X|Z)P(Y|Z)$. Equivalently: knowing $Y$ provides no additional information about $X$ once $Z$ is known.

## Relation to Graphical Models

Directed (BN): $X \perp \text{NonDescendants}(X) \mid \text{Parents}(X)$. Undirected (MRF): $X \perp Y \mid \text{Neighbors}(X)$ if no direct edge exists.

## Factorization

Enables tractable decomposition: $P(X_1, \ldots, X_n) = \prod_i P(X_i | \text{Pa}(X_i))$. Instead of exponentially many parameters, each conditional involves only the parent variables.

## Testing

Statistical tests: partial correlation (Gaussian data), conditional mutual information, chi-squared (discrete data). From graph structure: d-separation (directed) or graph separation (undirected), requiring no data.

## Faithfulness

A distribution is faithful to a graph if every conditional independence in the distribution is implied by the graph structure. Commonly assumed in structure learning.
