from setuptools import setup

setup(
    name='dasIT',
    version='1.0',
    packages=['dasIT', 'dasIT.src', 'dasIT.data', 'dasIT.features', 'dasIT.visualization'],
    url='https://github.com/luuleitner/dasIT',
    license='Apache License 2.0',
    author='Christoph Leitner',
    author_email='christoph.leitner@tugraz.at',
    description='plane-wave delay-and-sum beamformer'
)
