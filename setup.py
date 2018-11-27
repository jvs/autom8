#!/usr/bin/env python
import re
from setuptools import setup

with open('autom8/__init__.py', encoding='utf-8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open('README.md', encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='autom8',
    version=version,
    url='https://github.com/jvs/autom8',
    description='Python AutoML library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='John K. Von Seggern',
    author_email='john@machinate.com',
    packages=['autom8'],
    zip_safe=True,
    platforms='any',
    python_requires='>=3.6.0',
    install_requires=[
        'chardet>=3.0.4',
        'numpy>=1.14.4',
        'pandas>=0.23.4',
        'scikit-learn>=0.19.1',
        'scipy>=1.1.0',
    ],
    tests_require=['coverage', 'pytest>=3'],
    extras_require={
        'ml': [
            'lightgbm>=2.1.2',
            'xgboost>=0.80',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    keywords=['machine learning', 'autoML'],
)
