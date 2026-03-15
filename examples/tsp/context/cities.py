"""
20 fixed cities for the Traveling Salesman Problem.

Coordinates are on a 200x200 grid. The task is to find the shortest
closed tour visiting every city exactly once.
"""

import math

CITIES = [
    (60, 200),   # 0
    (180, 200),  # 1
    (80, 180),   # 2
    (140, 180),  # 3
    (20, 160),   # 4
    (100, 160),  # 5
    (200, 160),  # 6
    (140, 140),  # 7
    (40, 120),   # 8
    (100, 120),  # 9
    (180, 100),  # 10
    (60, 80),    # 11
    (120, 80),   # 12
    (180, 60),   # 13
    (100, 40),   # 14
    (40, 40),    # 15
    (140, 20),   # 16
    (20, 20),    # 17
    (200, 20),   # 18
    (120, 160),  # 19
]

NUM_CITIES = len(CITIES)


def tour_distance(tour):
    """
    Total Euclidean distance of a closed tour. Lower is better.

    Args:
        tour: list of city indices, a permutation of [0..NUM_CITIES-1]

    Returns:
        Total distance (float, rounded to 4 decimal places)
    """
    if len(tour) != NUM_CITIES:
        raise ValueError(f"Tour must visit all {NUM_CITIES} cities, got {len(tour)}")
    if set(tour) != set(range(NUM_CITIES)):
        raise ValueError("Tour must be a permutation of [0..19]")

    total = 0.0
    for i in range(NUM_CITIES):
        x1, y1 = CITIES[tour[i]]
        x2, y2 = CITIES[tour[(i + 1) % NUM_CITIES]]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(total, 4)
