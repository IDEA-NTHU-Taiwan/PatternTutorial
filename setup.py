import os
import platform

from setuptools import Command, find_packages, setup


class Clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        plat_name = platform.system()
        if plat_name == 'Windows':
            os.system(
                'rmdir /s /q "./build" "./dist" "./*.pyc" "./*.tgz" "./IDEAlib.egg-info"')
        elif plat_name in ['Linux', 'Darwin']:
            os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='IDEAlib',
    version='0.0.1.dev1',
    license='LICENSE',
    description='A Graph-based Pattern Representations',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/IDEA-NTHU-Taiwan/PatternTutorial.git',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'matplotlib',
        'networkx',
        'nltk',
        'pandas',
        'tqdm'
    ],
    python_requires='>=3.5.2',
    cmdclass={
        'clean': Clean
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: Linguistic'
    ),
)
