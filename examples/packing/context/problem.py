"""
Rectangle packing problem.

Pack 12 fixed-size rectangles into the smallest bounding box with no overlaps.
Each rectangle can be rotated 90 degrees. Score = bounding box area, with a
heavy penalty for overlapping rectangles.

Total rectangle area = 6975 (theoretical minimum bounding box area).
"""

RECTANGLES = [
    (40, 20),  # 0
    (30, 15),  # 1
    (50, 10),  # 2
    (25, 25),  # 3
    (35, 20),  # 4
    (20, 30),  # 5
    (45, 15),  # 6
    (15, 35),  # 7
    (30, 10),  # 8
    (20, 20),  # 9
    (40, 25),  # 10
    (10, 40),  # 11
]

NUM_RECTS = len(RECTANGLES)


def evaluate_packing(placements):
    """
    Evaluate a rectangle packing.

    Args:
        placements: list of (x, y, rotated) tuples.
            x, y = bottom-left corner (non-negative).
            rotated = True to swap width and height.

    Returns:
        score (int): bounding box area + 10000 * number_of_overlapping_pairs.
            Lower is better. A perfect packing has zero overlaps.
    """
    if len(placements) != NUM_RECTS:
        raise ValueError(f"Expected {NUM_RECTS} placements, got {len(placements)}")

    # Build placed rectangles as (x1, y1, x2, y2)
    rects = []
    max_x, max_y = 0, 0
    for i, (x, y, rotated) in enumerate(placements):
        w, h = RECTANGLES[i]
        if rotated:
            w, h = h, w
        rects.append((x, y, x + w, y + h))
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Count overlapping pairs
    overlaps = 0
    for i in range(NUM_RECTS):
        for j in range(i + 1, NUM_RECTS):
            ax1, ay1, ax2, ay2 = rects[i]
            bx1, by1, bx2, by2 = rects[j]
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                overlaps += 1

    area = max_x * max_y
    return area + 10000 * overlaps
