from pathlib import Path

import numpy as np


def main():
    obj = np.load("/home2/mingyang/projs/RoboticsDiffusionTransformer/cache/chunk_99/sample_99.npz")
    print(list(obj.keys()))

    mask = obj["state_vec_mask"]
    print(mask.dtype, mask.shape)

    print(np.nonzero(mask))


if __name__ == "__main__":
    main()
