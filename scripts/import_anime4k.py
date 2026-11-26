"""Convert the pinned MIT-licensed Anime4K shader into NumPy weights, offline."""
import argparse
import hashlib
from pathlib import Path
import re

import numpy as np

# Anime4K v4.0.1, glsl/Upscale/Anime4K_Upscale_CNN_x2_M.glsl
SHADER_SHA256 = "716e02098a68f0d648761f2b96b4dd139e1cb09b174bb369fca3aa34328fff7e"


def numbers(text):
    return np.fromstring(text, sep=",", dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    data = args.shader.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != SHADER_SHA256:
        parser.error("Shader differs from the published v4.0.1 coefficients.")
    blocks = data.decode("ascii").split("vec4 hook() {")[1:]
    arrays = {}
    for layer, block in enumerate(blocks[:7]):
        channels = 3 if layer == 0 else 8
        kernel = np.zeros((3, 3, channels, 4), dtype=np.float32)
        for values, group, x, y in re.findall(r"mat4\(([^)]+)\) \* go_(\d)\(([-.\d]+), ([-.\d]+)\)", block):
            start = int(group) * 4
            matrix = numbers(values).reshape(4, 4)
            # GLSL columns are input channels. The CPU uses row-vector products.
            kernel[int(float(y)) + 1, int(float(x)) + 1, start:start + 4] = matrix[:min(channels, 4)]
        arrays[f"k{layer}"] = kernel
        arrays[f"b{layer}"] = numbers(re.search(r"result \+= vec4\(([^)]+)\)", block)[1])
    head = re.findall(r"mat4\(([^)]+)\) \* g_(\d+);", blocks[7])
    assert [int(index) for _, index in head] == list(range(14))
    arrays["head"] = np.stack([numbers(values).reshape(4, 4) for values, _ in head]).reshape(7, 8, 4)
    arrays["bias"] = numbers(re.search(r"result \+= vec4\(([^)]+)\)", blocks[7])[1])
    np.savez_compressed(args.destination, **arrays)


if __name__ == "__main__":
    main()
