from setuptools import setup, find_packages

setup(
    name='TopoTrafficRL',
    version='1.0',
    description='Implementation of reinforcement learning based dense traffic awareness method.',
    url='https://github.com/Wei-Fan/TopoTrafficRL',
    author='Weifan Zhang',
    author_email='weifanz@umich.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],

    keywords='reinforcement learning agents',
    packages=find_packages(exclude=['docs', 'scripts', 'tests*']),
    install_requires=['gymnasium', 'numpy', 'pandas', 'numba', 'pygame', 'matplotlib', 'seaborn', 'six', 'docopt',
                      'torch>=1.2.0', 'tensorboardX', 'scipy'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
    },
    entry_points={
        'console_scripts': [],
    },
)

