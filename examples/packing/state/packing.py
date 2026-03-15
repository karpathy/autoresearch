"""
Rectangle placements. Each entry: (x, y, rotated).
Modify to minimize bounding box area with no overlaps.

See context/problem.py for rectangle sizes and the scoring function.
"""

# Initial: stacked vertically (valid, no overlaps, but very wasteful).
# Starting score: 13250 (bounding box 50 x 265, zero overlaps).
# Total rectangle area is 6975, so there's a lot of room to improve.
placements = [
    (0, 0, False),    # rect 0: 40x20
    (0, 20, False),   # rect 1: 30x15
    (0, 35, False),   # rect 2: 50x10
    (0, 45, False),   # rect 3: 25x25
    (0, 70, False),   # rect 4: 35x20
    (0, 90, False),   # rect 5: 20x30
    (0, 120, False),  # rect 6: 45x15
    (0, 135, False),  # rect 7: 15x35
    (0, 170, False),  # rect 8: 30x10
    (0, 180, False),  # rect 9: 20x20
    (0, 200, False),  # rect 10: 40x25
    (0, 225, False),  # rect 11: 10x40
]
