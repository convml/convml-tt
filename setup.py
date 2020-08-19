from setuptools import setup, find_packages

# note: version is maintained inside convml_tt/version.py
exec(open('convml_tt/version.py').read())

setup(
    name='convml_tt',
    packages=find_packages(exclude=['contrib', 'tests', 'docs']),
    version=__version__,
    description='Neural Network based study of convective organisation',
    author='Leif Denby',
    author_email='leifdenby@gmail.com',
    url='https://github.com/leifdenby/convml_tt',
    classifiers=[],
    install_requires=["xarray", "matplotlib", "fastai==1.0.53", "sklearn", "luigi"]
)
