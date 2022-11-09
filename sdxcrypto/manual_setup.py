import os
import sys
import pathlib
from pathlib import Path
import pkg_resources

from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log') 

def initial_setup():
    #install requirements 
    logger.info("Installing requirements")
    with pathlib.Path('../requirements.txt').open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]

    #install CLIP stuffs
    if not Path('src'):
        logger.info("Preparing CLIP")
        install_cmds = [
            # ['pip', 'install', 'ftfy', 'regex', 'tqdm', 'timm', 'fairscale', 'requests'],
            ['pip', 'install', '-e', 'git+https://github.com/moarshy/CLIP.git@main#egg=clip'],
            ['pip', 'install', '-e', 'git+https://github.com/moarshy/BLIP.git@lib#egg=blip'],
            ['git', 'clone', 'https://github.com/moarshy/clip-interrogator.git']
        ]
        for cmd in install_cmds:
            print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    else:
        logger.info("CLIP preparation exists. Please check")

    sys.path.append('src/blip')
    sys.path.append('src/clip')
    sys.path.append('clip-interrogator')

if __name__ == '__main__':
    initial_setup()