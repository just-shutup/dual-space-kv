from setuptools import setup, find_packages

setup(
    name="dual-space-kv",
    version="3.5.0",
    description="RoPE-Decoupled Streaming KV Cache Compression with First-Order Value Taylor Correction",
    author="Marley & Antigravity Research",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "transformers>=4.40.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
)
