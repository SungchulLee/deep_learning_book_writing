Great, let’s prove that identity carefully, step by step:

[
\log p_\theta(x_0)
==================

\mathcal{L}*{\text{ELBO}}(x_0)
+
\mathrm{KL}!\big(q(x*{1:T}\mid x_0),|,p_\theta(x_{1:T}\mid x_0)\big).
]

I’ll use short notation:

* (q(x_{1:T}\mid x_0)) = variational distribution
* (p_\theta(x_{1:T}\mid x_0)) = true posterior
* (x_{1:T}) means all latents (x_1,\dots,x_T)

---

### 1. Start from the **definition of KL**

By definition,
[
\mathrm{KL}!\big(q ,|, p_\theta(\cdot\mid x_0)\big)
===================================================

\mathbb{E}*q!\left[
\log \frac{q(x*{1:T}\mid x_0)}{p_\theta(x_{1:T}\mid x_0)}
\right].
]

Expand the log:

[
\mathrm{KL}(q|p_\theta)
=======================

\mathbb{E}*q\left[
\log q(x*{1:T}\mid x_0)
-----------------------

\log p_\theta(x_{1:T}\mid x_0)
\right].
]

Now use Bayes’ rule for the posterior:

[
p_\theta(x_{1:T}\mid x_0)
=========================

\frac{p_\theta(x_0, x_{1:T})}{p_\theta(x_0)}.
]

So

[
\log p_\theta(x_{1:T}\mid x_0)
==============================

\log p_\theta(x_0, x_{1:T}) - \log p_\theta(x_0).
]

Plug this into the KL expression.

---

### 2. Substitute Bayes’ rule into KL

[
\mathrm{KL}(q|p_\theta)
=======================

\mathbb{E}*q\left[
\log q(x*{1:T}\mid x_0)
-----------------------

\big(\log p_\theta(x_0, x_{1:T}) - \log p_\theta(x_0)\big)
\right].
]

Distribute the minus sign:

[
\mathrm{KL}(q|p_\theta)
=======================

\mathbb{E}*q\left[
\log q(x*{1:T}\mid x_0)
-----------------------

\log p_\theta(x_0, x_{1:T})
+
\log p_\theta(x_0)
\right].
]

Now split the expectation:

* (\log p_\theta(x_0)) is constant w.r.t. (q), so its expectation is just itself.

[
\mathrm{KL}(q|p_\theta)
=======================

\mathbb{E}*q\left[
\log q(x*{1:T}\mid x_0)
-----------------------

\log p_\theta(x_0, x_{1:T})
\right]
+
\log p_\theta(x_0).
]

Reorder terms:

[
\mathrm{KL}(q|p_\theta)
=======================

## \log p_\theta(x_0)

\mathbb{E}*q\left[
\log p*\theta(x_0, x_{1:T})
---------------------------

\log q(x_{1:T}\mid x_0)
\right].
]

---

### 3. Recognize the **ELBO** inside

By definition, the ELBO is

[
\mathcal{L}_{\text{ELBO}}(x_0)
==============================

\mathbb{E}*q\left[
\log p*\theta(x_0, x_{1:T})
---------------------------

\log q(x_{1:T}\mid x_0)
\right].
]

Compare this with the term in brackets above: it’s exactly the same.

So we have:

[
\mathrm{KL}(q|p_\theta)
=======================

## \log p_\theta(x_0)

\mathcal{L}_{\text{ELBO}}(x_0).
]

Now just rearrange:

[
\log p_\theta(x_0)
==================

\mathcal{L}*{\text{ELBO}}(x_0)
+
\mathrm{KL}(q|p*\theta).
]

Or, written fully:

[
\log p_\theta(x_0)
==================

\mathcal{L}*{\text{ELBO}}(x_0)
+
\mathrm{KL}\big(q(x*{1:T}\mid x_0),|,p_\theta(x_{1:T}\mid x_0)\big).
]

---

### 4. Intuition in one sentence

* The ELBO is the **expected log joint minus log (q)**.
* The KL is the **gap** between the ELBO and the true log-likelihood.
* That algebra above just makes this “gap = KL” fact explicit.

If you’d like, next I can:

* rewrite this in **compact “one-line” style** like in a paper, or
* give a **graphical picture**: ELBO as a lower bound with the gap labeled “KL”.
