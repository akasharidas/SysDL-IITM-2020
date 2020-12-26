"""
small channel:
    small img, small fil
    small img, large fil
    med img, small fil
    med img, large fil
    large img, small fil
    large img, large fil
    small img, oblong fil
    large img, oblong fil
    oblong img, small fil
    oblong img, large fil
large channel:
    small img, small fil
    small img, large fil
    med img, small fil
    med img, large fil
    large img, small fil
    large img, large fil
    small img, oblong fil
    large img, oblong fil
    oblong img, small fil
    oblong img, large fil
"""

configs = [
    (3, 32, 32, 3, 3, 3),
    (3, 32, 32, 3, 7, 7),
    (3, 64, 64, 3, 3, 3),
    (3, 64, 64, 3, 7, 7),
    (3, 224, 224, 3, 3, 3),
    (3, 224, 224, 3, 7, 7),
    (3, 224, 224, 3, 3, 7),
    (3, 32, 32, 3, 3, 7),
    (3, 32, 224, 3, 3, 3),
    (3, 32, 224, 3, 7, 7),
    (32, 32, 32, 3, 3, 3),
    (32, 32, 32, 3, 7, 7),
    (32, 64, 64, 3, 3, 3),
    (32, 64, 64, 3, 7, 7),
    (32, 224, 224, 3, 3, 3),
    (32, 224, 224, 3, 7, 7),
    (32, 32, 32, 3, 3, 7),
    (32, 224, 224, 3, 3, 7),
    (32, 32, 224, 3, 3, 3),
    (32, 32, 224, 3, 7, 7),
]
