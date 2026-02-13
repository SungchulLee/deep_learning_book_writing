# ========================================================
# logistic_regression/main.py
# ========================================================
import logistic_regression as lor

# Parse arguments and set seed
cfg = lor.parse_args()
lor.set_seed(seed=cfg.seed)

# Test all drug responses
print("\n=== Testing All Drug Responses ===")
lor.test_all_responses(seed=cfg.seed, lr=cfg.lr, epochs=cfg.epochs)

# Run other benchmarks
print("\n=== Other Benchmarks ===")
lor.benchmark_lr()
print()
lor.benchmark_epochs()
print()
lor.compare_sklearn()