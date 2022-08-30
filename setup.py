from setuptools import setup, find_packages

setup(
    name='mango_python',
    version='0.0.3',
    description='Personal python package',
    
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    
    packages=find_packages(
        include=['mango_python']
    )
)
