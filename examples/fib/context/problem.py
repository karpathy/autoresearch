"""
Fibonacci performance optimization.

The task is to make fib(n) as fast as possible while remaining correct.
The scoring function validates correctness against known values, then
benchmarks fib(20) and returns the median wall-clock time in seconds.

Known Fibonacci values (0-indexed):
    fib(0)  = 0
    fib(1)  = 1
    fib(10) = 55
    fib(20) = 6765

Rules:
    - fib(n) must return the correct value for any non-negative integer n
    - No hardcoding return values — the function must compute them
    - No reading from the scoring directory
    - The function signature must remain: def fib(n) -> int
"""

BENCHMARK_N = 20
EXPECTED_FIB_20 = 6765

CORRECTNESS_CASES = [
    (0, 0),
    (1, 1),
    (2, 1),
    (5, 5),
    (10, 55),
    (20, 6765),
]
