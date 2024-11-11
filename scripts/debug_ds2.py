import numpy as np
from PIL import Image


def main():
    obj = np.load("libero/datasets/calvin_format/validation/episode_0000000.npz")

    Image.fromarray(obj["rgb_static"]).save("outputs/rgb_static.png")
    Image.fromarray(obj["rgb_gripper"]).save("outputs/rgb_gripper.png")


if __name__ == "__main__":
    main()
