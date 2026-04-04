"""
Code Solver Benchmark — Evaluation Harness
DO NOT MODIFY (agent reads only)

30 algorithmic problems (Easy/Medium/Hard) with hidden test cases.
Evaluates solve.py and reports pass_rate as the primary metric.

Usage:
    uv run prepare.py            # full evaluation
    uv run prepare.py --quick    # first failure per problem only (faster debugging)
"""

import importlib
import sys
import time
import traceback
import math
from typing import Any

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TIME_BUDGET = 60  # seconds — generous; most runs finish in <5s
TIMEOUT_PER_PROBLEM = 5  # seconds per problem

# ──────────────────────────────────────────────────────────────
# Problem Definitions
# ──────────────────────────────────────────────────────────────

def _sort_inner(lst):
    """Sort list of lists for order-independent comparison."""
    return sorted([sorted(x) for x in lst])

PROBLEMS = [
    # ── Easy (1-10) ──────────────────────────────────────────
    {
        "id": "two_sum",
        "difficulty": "easy",
        "description": "Given list nums and int target, return sorted [i, j] of two indices whose values sum to target. Exactly one solution exists.",
        "tests": [
            {"args": ([2, 7, 11, 15], 9), "expected": [0, 1]},
            {"args": ([3, 2, 4], 6), "expected": [1, 2]},
            {"args": ([3, 3], 6), "expected": [0, 1]},
            {"args": ([1, 5, 3, 7, 2], 9), "expected": [1, 3]},
            {"args": ([-1, -2, -3, -4, -5], -8), "expected": [2, 4]},
        ],
    },
    {
        "id": "is_palindrome",
        "difficulty": "easy",
        "description": "Return True if string is a palindrome considering only alphanumeric chars, case-insensitive.",
        "tests": [
            {"args": ("A man, a plan, a canal: Panama",), "expected": True},
            {"args": ("race a car",), "expected": False},
            {"args": ("",), "expected": True},
            {"args": (" ",), "expected": True},
            {"args": ("0P",), "expected": False},
            {"args": ("aa",), "expected": True},
            {"args": (".,",), "expected": True},
        ],
    },
    {
        "id": "fizzbuzz",
        "difficulty": "easy",
        "description": "Return list of strings for 1..n: 'FizzBuzz' if div by 15, 'Fizz' if div by 3, 'Buzz' if div by 5, else str(i).",
        "tests": [
            {"args": (5,), "expected": ["1", "2", "Fizz", "4", "Buzz"]},
            {"args": (15,), "expected": ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]},
            {"args": (1,), "expected": ["1"]},
            {"args": (3,), "expected": ["1", "2", "Fizz"]},
        ],
    },
    {
        "id": "fibonacci",
        "difficulty": "easy",
        "description": "Return the n-th Fibonacci number (0-indexed). fib(0)=0, fib(1)=1, fib(2)=1, ...",
        "tests": [
            {"args": (0,), "expected": 0},
            {"args": (1,), "expected": 1},
            {"args": (2,), "expected": 1},
            {"args": (10,), "expected": 55},
            {"args": (20,), "expected": 6765},
            {"args": (30,), "expected": 832040},
        ],
    },
    {
        "id": "max_profit",
        "difficulty": "easy",
        "description": "Given list of stock prices by day, return max profit from one buy then sell. Return 0 if no profit possible.",
        "tests": [
            {"args": ([7, 1, 5, 3, 6, 4],), "expected": 5},
            {"args": ([7, 6, 4, 3, 1],), "expected": 0},
            {"args": ([1, 2],), "expected": 1},
            {"args": ([2, 4, 1],), "expected": 2},
            {"args": ([3, 3, 3],), "expected": 0},
            {"args": ([1],), "expected": 0},
        ],
    },
    {
        "id": "valid_parentheses",
        "difficulty": "easy",
        "description": "Return True if string of '()[]{}' has valid matching brackets.",
        "tests": [
            {"args": ("()",), "expected": True},
            {"args": ("()[]{}",), "expected": True},
            {"args": ("(]",), "expected": False},
            {"args": ("([)]",), "expected": False},
            {"args": ("{[]}",), "expected": True},
            {"args": ("",), "expected": True},
            {"args": ("]",), "expected": False},
            {"args": ("(((",), "expected": False},
        ],
    },
    {
        "id": "merge_sorted",
        "difficulty": "easy",
        "description": "Merge two sorted lists into one sorted list.",
        "tests": [
            {"args": ([1, 3, 5], [2, 4, 6]), "expected": [1, 2, 3, 4, 5, 6]},
            {"args": ([], [1, 2, 3]), "expected": [1, 2, 3]},
            {"args": ([1], []), "expected": [1]},
            {"args": ([], []), "expected": []},
            {"args": ([1, 1, 1], [1, 1]), "expected": [1, 1, 1, 1, 1]},
            {"args": ([1, 2, 3], [4, 5, 6]), "expected": [1, 2, 3, 4, 5, 6]},
        ],
    },
    {
        "id": "remove_duplicates",
        "difficulty": "easy",
        "description": "Return a new sorted list with duplicates removed from input list.",
        "tests": [
            {"args": ([1, 1, 2],), "expected": [1, 2]},
            {"args": ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4],), "expected": [0, 1, 2, 3, 4]},
            {"args": ([],), "expected": []},
            {"args": ([1],), "expected": [1]},
            {"args": ([3, 1, 2, 1, 3],), "expected": [1, 2, 3]},
        ],
    },
    {
        "id": "count_chars",
        "difficulty": "easy",
        "description": "Return dict mapping each character to its count in the string.",
        "tests": [
            {"args": ("hello",), "expected": {"h": 1, "e": 1, "l": 2, "o": 1}},
            {"args": ("",), "expected": {}},
            {"args": ("aaa",), "expected": {"a": 3}},
            {"args": ("abcabc",), "expected": {"a": 2, "b": 2, "c": 2}},
        ],
    },
    {
        "id": "reverse_words",
        "difficulty": "easy",
        "description": "Reverse the order of words in a string. Words are separated by spaces. Remove leading/trailing/extra spaces.",
        "tests": [
            {"args": ("the sky is blue",), "expected": "blue is sky the"},
            {"args": ("  hello world  ",), "expected": "world hello"},
            {"args": ("a good   example",), "expected": "example good a"},
            {"args": ("single",), "expected": "single"},
            {"args": ("",), "expected": ""},
        ],
    },

    # ── Medium (11-20) ───────────────────────────────────────
    {
        "id": "max_subarray",
        "difficulty": "medium",
        "description": "Return maximum sum of a contiguous subarray (Kadane's algorithm).",
        "tests": [
            {"args": ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), "expected": 6},
            {"args": ([1],), "expected": 1},
            {"args": ([5, 4, -1, 7, 8],), "expected": 23},
            {"args": ([-1],), "expected": -1},
            {"args": ([-2, -1],), "expected": -1},
            {"args": ([-1, -2, -3],), "expected": -1},
        ],
    },
    {
        "id": "group_anagrams",
        "difficulty": "medium",
        "description": "Group anagrams together. Return list of groups; each group sorted, groups sorted by first element.",
        "tests": [
            {
                "args": (["eat", "tea", "tan", "ate", "nat", "bat"],),
                "expected": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]],
                "compare": _sort_inner,
            },
            {"args": ([""],), "expected": [[""]], "compare": _sort_inner},
            {"args": (["a"],), "expected": [["a"]], "compare": _sort_inner},
        ],
    },
    {
        "id": "longest_common_prefix",
        "difficulty": "medium",
        "description": "Find the longest common prefix string amongst a list of strings. Return '' if none.",
        "tests": [
            {"args": (["flower", "flow", "flight"],), "expected": "fl"},
            {"args": (["dog", "racecar", "car"],), "expected": ""},
            {"args": (["interspecies", "interstellar", "interstate"],), "expected": "inters"},
            {"args": (["a"],), "expected": "a"},
            {"args": ([],), "expected": ""},
            {"args": (["", "b"],), "expected": ""},
        ],
    },
    {
        "id": "binary_search",
        "difficulty": "medium",
        "description": "Return index of target in sorted list, or -1 if not found.",
        "tests": [
            {"args": ([-1, 0, 3, 5, 9, 12], 9), "expected": 4},
            {"args": ([-1, 0, 3, 5, 9, 12], 2), "expected": -1},
            {"args": ([], 5), "expected": -1},
            {"args": ([5], 5), "expected": 0},
            {"args": ([1, 2, 3, 4, 5], 1), "expected": 0},
            {"args": ([1, 2, 3, 4, 5], 5), "expected": 4},
        ],
    },
    {
        "id": "three_sum",
        "difficulty": "medium",
        "description": "Return all unique triplets [a,b,c] from nums where a+b+c=0. Each triplet sorted, result sorted.",
        "tests": [
            {"args": ([-1, 0, 1, 2, -1, -4],), "expected": [[-1, -1, 2], [-1, 0, 1]]},
            {"args": ([0, 1, 1],), "expected": []},
            {"args": ([0, 0, 0],), "expected": [[0, 0, 0]]},
            {"args": ([0, 0, 0, 0],), "expected": [[0, 0, 0]]},
            {"args": ([-2, 0, 1, 1, 2],), "expected": [[-2, 0, 2], [-2, 1, 1]]},
        ],
    },
    {
        "id": "container_water",
        "difficulty": "medium",
        "description": "Given list of heights, find two lines that form a container holding the most water. Return max area.",
        "tests": [
            {"args": ([1, 8, 6, 2, 5, 4, 8, 3, 7],), "expected": 49},
            {"args": ([1, 1],), "expected": 1},
            {"args": ([4, 3, 2, 1, 4],), "expected": 16},
            {"args": ([1, 2, 1],), "expected": 2},
        ],
    },
    {
        "id": "coin_change",
        "difficulty": "medium",
        "description": "Return fewest coins needed to make amount. Return -1 if impossible. coins is list of denominations.",
        "tests": [
            {"args": ([1, 5, 10, 25], 30), "expected": 2},
            {"args": ([2], 3), "expected": -1},
            {"args": ([1], 0), "expected": 0},
            {"args": ([1, 2, 5], 11), "expected": 3},
            {"args": ([2, 5, 10, 1], 27), "expected": 4},
            {"args": ([186, 419, 83, 408], 6249), "expected": 20},
        ],
    },
    {
        "id": "product_except_self",
        "difficulty": "medium",
        "description": "Return list where each element is the product of all other elements. No division allowed. O(n) time.",
        "tests": [
            {"args": ([1, 2, 3, 4],), "expected": [24, 12, 8, 6]},
            {"args": ([-1, 1, 0, -3, 3],), "expected": [0, 0, 9, 0, 0]},
            {"args": ([2, 3],), "expected": [3, 2]},
            {"args": ([0, 0],), "expected": [0, 0]},
        ],
    },
    {
        "id": "longest_substring_no_repeat",
        "difficulty": "medium",
        "description": "Return length of the longest substring without repeating characters.",
        "tests": [
            {"args": ("abcabcbb",), "expected": 3},
            {"args": ("bbbbb",), "expected": 1},
            {"args": ("pwwkew",), "expected": 3},
            {"args": ("",), "expected": 0},
            {"args": ("au",), "expected": 2},
            {"args": ("dvdf",), "expected": 3},
            {"args": ("abcdef",), "expected": 6},
        ],
    },
    {
        "id": "rotate_matrix",
        "difficulty": "medium",
        "description": "Rotate an NxN matrix 90 degrees clockwise. Return the new matrix.",
        "tests": [
            {"args": ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), "expected": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]},
            {"args": ([[1, 2], [3, 4]],), "expected": [[3, 1], [4, 2]]},
            {"args": ([[1]],), "expected": [[1]]},
            {
                "args": ([[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]],),
                "expected": [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]],
            },
        ],
    },

    # ── Hard (21-30) ─────────────────────────────────────────
    {
        "id": "lcs_length",
        "difficulty": "hard",
        "description": "Return length of the Longest Common Subsequence of two strings.",
        "tests": [
            {"args": ("abcde", "ace"), "expected": 3},
            {"args": ("abc", "abc"), "expected": 3},
            {"args": ("abc", "def"), "expected": 0},
            {"args": ("", "abc"), "expected": 0},
            {"args": ("oxcpqrsvwf", "shmtulqrypy"), "expected": 2},
            {"args": ("abcba", "abcbcba"), "expected": 5},
        ],
    },
    {
        "id": "edit_distance",
        "difficulty": "hard",
        "description": "Return minimum edit distance (insert, delete, replace) between two strings.",
        "tests": [
            {"args": ("horse", "ros"), "expected": 3},
            {"args": ("intention", "execution"), "expected": 5},
            {"args": ("", ""), "expected": 0},
            {"args": ("a", ""), "expected": 1},
            {"args": ("abc", "abc"), "expected": 0},
            {"args": ("kitten", "sitting"), "expected": 3},
        ],
    },
    {
        "id": "longest_increasing_subsequence",
        "difficulty": "hard",
        "description": "Return length of the longest strictly increasing subsequence.",
        "tests": [
            {"args": ([10, 9, 2, 5, 3, 7, 101, 18],), "expected": 4},
            {"args": ([0, 1, 0, 3, 2, 3],), "expected": 4},
            {"args": ([7, 7, 7, 7, 7],), "expected": 1},
            {"args": ([],), "expected": 0},
            {"args": ([1],), "expected": 1},
            {"args": ([1, 3, 6, 7, 9, 4, 10, 5, 6],), "expected": 6},
        ],
    },
    {
        "id": "knapsack",
        "difficulty": "hard",
        "description": "0/1 knapsack: given lists weights and values of same length, and capacity, return max total value.",
        "tests": [
            {"args": ([1, 2, 3], [6, 10, 12], 5), "expected": 22},
            {"args": ([2, 3, 4, 5], [3, 4, 5, 6], 5), "expected": 7},
            {"args": ([], [], 10), "expected": 0},
            {"args": ([10], [100], 5), "expected": 0},
            {"args": ([1, 1, 1], [10, 20, 30], 2), "expected": 50},
            {"args": ([3, 4, 2], [4, 5, 3], 7), "expected": 9},
        ],
    },
    {
        "id": "spiral_order",
        "difficulty": "hard",
        "description": "Return elements of an MxN matrix in spiral order (clockwise from top-left).",
        "tests": [
            {"args": ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), "expected": [1, 2, 3, 6, 9, 8, 7, 4, 5]},
            {"args": ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],), "expected": [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]},
            {"args": ([[1]],), "expected": [1]},
            {"args": ([[1, 2], [3, 4]],), "expected": [1, 2, 4, 3]},
            {"args": ([[1], [2], [3]],), "expected": [1, 2, 3]},
        ],
    },
    {
        "id": "word_break",
        "difficulty": "hard",
        "description": "Return True if string s can be segmented into space-separated sequence of dictionary words.",
        "tests": [
            {"args": ("leetcode", ["leet", "code"]), "expected": True},
            {"args": ("applepenapple", ["apple", "pen"]), "expected": True},
            {"args": ("catsandog", ["cats", "dog", "sand", "and", "cat"]), "expected": False},
            {"args": ("", ["a"]), "expected": True},
            {"args": ("a", []), "expected": False},
            {"args": ("aaaaaaa", ["aaa", "aaaa"]), "expected": True},
            {"args": ("goalspecial", ["go", "goal", "goals", "special"]), "expected": True},
        ],
    },
    {
        "id": "decode_ways",
        "difficulty": "hard",
        "description": "Count ways to decode digit string where '1'='A', '2'='B', ..., '26'='Z'. Return count.",
        "tests": [
            {"args": ("12",), "expected": 2},
            {"args": ("226",), "expected": 3},
            {"args": ("06",), "expected": 0},
            {"args": ("0",), "expected": 0},
            {"args": ("10",), "expected": 1},
            {"args": ("27",), "expected": 1},
            {"args": ("11106",), "expected": 2},
            {"args": ("111111",), "expected": 13},
        ],
    },
    {
        "id": "largest_rectangle_histogram",
        "difficulty": "hard",
        "description": "Given list of bar heights, find the largest rectangular area in the histogram.",
        "tests": [
            {"args": ([2, 1, 5, 6, 2, 3],), "expected": 10},
            {"args": ([2, 4],), "expected": 4},
            {"args": ([1],), "expected": 1},
            {"args": ([1, 1],), "expected": 2},
            {"args": ([5, 4, 3, 2, 1],), "expected": 9},
            {"args": ([1, 2, 3, 4, 5],), "expected": 9},
            {"args": ([3, 6, 5, 7, 4, 8, 1, 0],), "expected": 20},
        ],
    },
    {
        "id": "median_sorted_arrays",
        "difficulty": "hard",
        "description": "Find median of two sorted arrays. Return float.",
        "tests": [
            {"args": ([1, 3], [2]), "expected": 2.0},
            {"args": ([1, 2], [3, 4]), "expected": 2.5},
            {"args": ([], [1]), "expected": 1.0},
            {"args": ([2], []), "expected": 2.0},
            {"args": ([1, 2, 3], [4, 5, 6]), "expected": 3.5},
            {"args": ([1, 1, 1], [1, 1, 1]), "expected": 1.0},
        ],
    },
    {
        "id": "serialize_deserialize_tree",
        "difficulty": "hard",
        "description": "Implement serialize(root) and deserialize(data). root is a nested list: [val, left, right] or None. serialize returns a string, deserialize reconstructs the tree. deserialize(serialize(tree)) must equal original tree.",
        "tests": [
            {"args": ([1, [2, None, None], [3, [4, None, None], [5, None, None]]],), "expected": [1, [2, None, None], [3, [4, None, None], [5, None, None]]], "special": "roundtrip"},
            {"args": (None,), "expected": None, "special": "roundtrip"},
            {"args": ([1, None, None],), "expected": [1, None, None], "special": "roundtrip"},
            {"args": ([1, [2, [3, None, None], None], None],), "expected": [1, [2, [3, None, None], None], None], "special": "roundtrip"},
        ],
    },
]

# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def _run_test(func, test, problem_id):
    """Run a single test case with timeout protection."""
    args = test["args"]
    expected = test["expected"]
    special = test.get("special", None)

    try:
        if special == "roundtrip":
            # For serialize/deserialize: test roundtrip
            import solve
            serialized = solve.serialize(args[0])
            if not isinstance(serialized, str):
                return False, f"serialize must return str, got {type(serialized).__name__}"
            result = solve.deserialize(serialized)
            if result != expected:
                return False, f"roundtrip failed: got {result!r}"
            return True, None
        else:
            result = func(*args)

        # Custom comparator if provided
        compare = test.get("compare", None)
        if compare:
            if compare(result) != compare(expected):
                return False, f"got {result!r}, expected {expected!r}"
        elif isinstance(expected, float):
            if not isinstance(result, (int, float)):
                return False, f"got {result!r}, expected float"
            if abs(result - expected) > 1e-5:
                return False, f"got {result}, expected {expected}"
        else:
            if result != expected:
                return False, f"got {result!r}, expected {expected!r}"
        return True, None

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def evaluate(quick=False):
    """Run all problems against solve.py. Return (passed, total, details)."""
    try:
        import solve
        importlib.reload(solve)  # pick up latest changes
    except Exception as e:
        print(f"FATAL: Cannot import solve.py: {e}")
        return 0, len(PROBLEMS), []

    start = time.time()
    passed = 0
    total = len(PROBLEMS)
    details = []

    easy_pass = 0
    easy_total = 0
    medium_pass = 0
    medium_total = 0
    hard_pass = 0
    hard_total = 0

    for problem in PROBLEMS:
        pid = problem["id"]
        diff = problem["difficulty"]

        if diff == "easy":
            easy_total += 1
        elif diff == "medium":
            medium_total += 1
        else:
            hard_total += 1

        # Check if function exists
        if problem.get("tests", [{}])[0].get("special") == "roundtrip":
            has_func = hasattr(solve, "serialize") and hasattr(solve, "deserialize")
        else:
            has_func = hasattr(solve, pid)

        if not has_func:
            status = "SKIP"
            details.append((pid, diff, status, "not implemented"))
            print(f"  {status:5s} [{diff:6s}] {pid}: not implemented")
            continue

        func = getattr(solve, pid, None)
        all_pass = True
        fail_msg = None

        for i, test in enumerate(problem["tests"]):
            ok, msg = _run_test(func, test, pid)
            if not ok:
                all_pass = False
                fail_msg = f"test {i}: {msg}"
                if quick:
                    break

        if all_pass:
            passed += 1
            status = "PASS"
            if diff == "easy":
                easy_pass += 1
            elif diff == "medium":
                medium_pass += 1
            else:
                hard_pass += 1
            details.append((pid, diff, status, ""))
            print(f"  {status:5s} [{diff:6s}] {pid}")
        else:
            status = "FAIL"
            details.append((pid, diff, status, fail_msg))
            print(f"  {status:5s} [{diff:6s}] {pid}: {fail_msg}")

    elapsed = time.time() - start
    pass_rate = passed / total if total > 0 else 0.0

    # Print summary in autoresearch-compatible format
    print(f"\n---")
    print(f"pass_rate:    {pass_rate:.6f}")
    print(f"solved:       {passed}/{total}")
    print(f"easy:         {easy_pass}/{easy_total}")
    print(f"medium:       {medium_pass}/{medium_total}")
    print(f"hard:         {hard_pass}/{hard_total}")
    print(f"elapsed_sec:  {elapsed:.1f}")

    return passed, total, details


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    evaluate(quick=quick)
