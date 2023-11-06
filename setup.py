# Copyright 2023 Pavlo Bashmakov

"""setup.py file for ptudes-lab"""
from setuptools import find_namespace_packages
from setuptools import setup


__version__ = '0.0.1'


with open('README.md', encoding='utf-8') as f:
    _long_description = f.read()

setup(
    name='ptudes-lab',
    version=__version__,
    description=
    'Single/multi-lidar systems experiments with calibration, visualization and localization',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='Pavlo Bashmakov',
    author_email='157482+bexcite@users.noreply.github.com',
    python_requires='>=3.10',
    packages=find_namespace_packages(where='src',
                                     include=['ptudes.*', 'ouster.*']),
    package_dir={'': 'src'},
    install_requires=[
        'numpy >= 1.20',
        'ouster-sdk >= 0.10.0',
        # 'matplotlib>=3.7.1',
        # 'Pillow>=9.4.0',
        # 'tqdm>=4.65.0',
    ],
    entry_points={'console_scripts': ['ptudes=ptudes.cli.run:main']},
    url='https://github.com/bexcite/ptudes-lab',
    license='MIT',
)
