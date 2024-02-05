# Copyright 2023 Pavlo Bashmakov

"""setup.py file for ptudes-lab"""
from setuptools import find_namespace_packages
from setuptools import setup


__version__ = '0.0.3'

setup(
    name='ptudes-lab',
    version=__version__,
    description='Ptudes lab: odometry, SLAM and visualization experiments',
    author='Pavlo Bashmakov',
    author_email='157482+bexcite@users.noreply.github.com',
    python_requires='>=3.8, <3.12',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy >= 1.20',
        'ouster-sdk >= 0.10.0',
        'rosbags >= 0.9.16',
        'matplotlib >= 3.5.3',
        'kiss-icp >= 0.2.9, <3'
    ],
    entry_points={'console_scripts': ['ptudes=ptudes.cli.run:main']},
    url='https://github.com/bexcite/ptudes-lab',
    license='MIT',
)
