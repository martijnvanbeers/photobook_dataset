from setuptools import setup

setup(
    name='photobook',
    packages=['photobook'],
    include_package_data=True,
    install_requires=[
        'torch', 'numpy', 'nltk',
    ],
)
