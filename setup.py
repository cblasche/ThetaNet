from setuptools import setup, find_packages

setup(
    name="ThetaNet",
    version="0.1",
    packages=find_packages(),

    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'time'],

    author=u"Christian Blasche",
    author_email="c.b@mail.de",
    description="A python library for simulating networks of theta neurons.",
    license="GPL v2",
    keywords=["theta neuron", "network"],
    url="https://github.com/cblasche/ThetaNet"
)
