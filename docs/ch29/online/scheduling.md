# Online Scheduling

In many real-world systems, tasks arrive over time and must be assigned to machines immediately without knowledge of future arrivals. A data center receives jobs from users and must assign them to servers; a GPU cluster receives training jobs and must allocate resources. **Online scheduling** studies how to make these irrevocable assignment decisions while minimizing objectives like the total completion time (makespan) or total weighted flow time, and competitive analysis provides the framework for measuring how well these decisions perform.

## Problem Formulation

In the **online load balancing** (or online makespan minimization) problem:

- There are $m$ identical machines.
- Jobs $j_1, j_2, \ldots, j_n$ arrive one at a time, each with processing time $p_j$.
- Upon arrival, job $j$ must be irrevocably assigned to one of the $m$ machines.
- The **makespan** is the maximum load across all machines:

$$
C_{\max} = \max_{i=1}^{m} \sum_{j \in M_i} p_j
$$

where $M_i$ is the set of jobs assigned to machine $i$.

The goal is to minimize $C_{\max}$ relative to the offline optimal makespan $C_{\max}^*$.

## Greedy Algorithms

### List Scheduling (Graham, 1966)

The simplest strategy assigns each arriving job to the **least loaded machine**:

$$
\text{Assign job } j \text{ to machine } i^* = \arg\min_{i} L_i
$$

where $L_i$ is the current load on machine $i$.

**Theorem (Graham).** List scheduling is $(2 - 1/m)$-competitive.

*Proof.* Let $j^*$ be the last job to finish on the most loaded machine. At the time $j^*$ was assigned, that machine had the minimum load, so its load before $j^*$ was at most the average load:

$$
L_{i^*} - p_{j^*} \leq \frac{1}{m} \sum_{j} p_j \leq C_{\max}^*
$$

Since $p_{j^*} \leq C_{\max}^*$ (no single job can exceed the optimal makespan), we get:

$$
C_{\max} = L_{i^*} \leq C_{\max}^* + C_{\max}^* - \frac{C_{\max}^*}{m} = \left(2 - \frac{1}{m}\right) C_{\max}^*
$$

$\square$

!!! note "Tightness"
    The bound $2 - 1/m$ is tight: an adversary can construct a sequence where list scheduling achieves exactly this ratio. However, it can be improved with additional information.

### LPT (Longest Processing Time First)

If jobs can be sorted offline by decreasing processing time before assignment, the **LPT** rule achieves a better ratio:

$$
C_{\max}^{\text{LPT}} \leq \left(\frac{4}{3} - \frac{1}{3m}\right) C_{\max}^*
$$

LPT is an offline algorithm, but it motivates online variants where large jobs are prioritized.

## Related Machines

When machines have different speeds $s_1, s_2, \ldots, s_m$, job $j$ takes time $p_j / s_i$ on machine $i$. The competitive ratio depends on the speed ratios:

$$
\text{Greedy is } \Theta(\log m)\text{-competitive for related machines}
$$

More sophisticated algorithms using doubling techniques achieve $O(1)$-competitive ratios for related machines.

## Weighted Completion Time

Another common objective is minimizing the total weighted completion time:

$$
\sum_{j} w_j \cdot C_j
$$

where $w_j$ is the weight (priority) of job $j$ and $C_j$ is its completion time. Online algorithms for this objective use the **Weighted Shortest Job First (WSJF)** rule, processing jobs in decreasing order of $w_j / p_j$.

## Online Scheduling with Preemption

If jobs can be **preempted** (interrupted and resumed later), online algorithms gain significant power. Preemptive scheduling allows the algorithm to correct past decisions:

- **SRPT (Shortest Remaining Processing Time)** is 1-competitive for minimizing total completion time on a single machine with preemption.
- For makespan on $m$ machines, preemption does not improve the competitive ratio.

## Lower Bounds

**Theorem.** No deterministic online algorithm for makespan minimization on $m$ identical machines can achieve a competitive ratio better than $1.5$ for $m \geq 3$.

For $m = 2$, the tight bound is $3/2$. For general $m$, the lower bound is approximately $1.88$.

## Connection to Deep Learning

Online scheduling appears naturally in deep learning infrastructure:

- **Job scheduling on GPU clusters**: training jobs arrive continuously and must be assigned to available GPUs. The competitive ratio of the scheduling algorithm directly impacts cluster utilization.
- **Pipeline parallelism**: in model parallelism, micro-batches are scheduled across pipeline stages, and load imbalance creates pipeline bubbles analogous to makespan overhead.
- **Dynamic batching**: in inference serving, incoming requests must be batched and assigned to devices, balancing latency and throughput in an online manner.

## Summary

Online scheduling studies how to assign jobs to machines without knowing future arrivals. Graham's list scheduling achieves the tight deterministic ratio of $2 - 1/m$ for identical machines, while LPT improves this offline to $4/3 - 1/(3m)$. Preemption, randomization, and resource augmentation provide avenues for improvement. The problem has direct applications in GPU cluster management and distributed training systems.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
