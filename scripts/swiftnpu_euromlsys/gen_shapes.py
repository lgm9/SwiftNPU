import random
from pathlib import Path

OUT_FILE = Path("results/shapes.txt")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Format:
#   N M K c     -> matmul: (N x M) @ (M x K), scheduled on c cores

# BERT
# sizes = [
#     (256, 3072, 768, 64),
#     (128, 3072, 768, 32),
#     (64, 3072, 768, 16),
#     (32, 3072, 768, 8),
# ]

# ResNet
# sizes = [
#     (3136, 256, 512, 64),
#     (3136, 128, 512, 32),
#     (3136, 64, 512, 16),
#     (3136, 32, 512, 8),
# ]


sizes = [
    (256, 3072, 768, 64),
    (128, 3072, 768, 32),
    (64, 3072, 768, 16),
    (32, 3072, 768, 8),
    (3136, 256, 512, 64),
    (3136, 128, 512, 32),
    (3136, 64, 512, 16),
    (3136, 32, 512, 8),
]

with open(OUT_FILE, "w") as f:
    for _ in range(500):
        N, M, K, c = random.choice(sizes)
        f.write(f"{N} {M} {K} {c}\n")
