from setuptools import setup, find_packages

setup(
    name="mdt",  # Replace with your project name
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List any dependencies here, for example:
        "pytorch-lightning==1.9.5",
        "pyhash",
        "torchsde~=0.2.6",
    ],
    entry_points={
        # Optional, if you have a command-line script
        # "console_scripts": [
        #     "my_command=my_package.module:function",
        # ],
    },
)
