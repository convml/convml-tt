from setuptools import setup, find_packages

setup(
    name='convml_tt',
    packages=find_packages(exclude=['contrib', 'tests', 'docs']),
    version='0.3.0',
    description='Neural Network based study of convective organisation',
    author='Leif Denby',
    author_email='leifdenby@gmail.com',
    url='https://github.com/leifdenby/convml_tt',
    classifiers=[],
    install_requires=["xarray", "matplotlib", "fastai==1.0.42", "sklearn"]
)
