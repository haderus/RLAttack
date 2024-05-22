import sys
import setuptools

if sys.version_info < (3, 8):
    sys.exit('Python>=3.8 is required by ReLAttack.')

setuptools.setup(
    name="ReLAttack",
    version='0.1.0',
    author=("anonymous"),
    description="RL",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='RL',
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)