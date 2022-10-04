import pathlib

import pkg_resources
import setuptools

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setuptools.setup(name='sdxcrypto',
                version='0.0.1',
                description='Algovera Stable Diffusion X Crypto',
                author='Algovera AI',
                author_email='hello@algovera.ai',
                url='https://www.algovera.ai/',
                install_requires=install_requires,
                )