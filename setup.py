from setuptools import setup

from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        check_call("cd ./externas/devernay_1.0 && make clean && make", shell=True)
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        check_call("cd ./externas/devernay_1.0 && make clean && make", shell=True)
        install.run(self)

setup(
    name='cross_section_tree_ring_detection',
    version='1.0.0',
    description='Cross section tree ring detection method over RGB images',
    url='https://github.com/hmarichal93/cstrd_ipol',
    author='Henry Marichal',
    author_email='hmarichal93@gmail.com',
    license='MIT',
    packages=['cross_section_tree_ring_detection'],
    install_requires=['numpy==1.26.1',
        'matplotlib==3.8.0',
        'opencv-python==4.8.1.78',
        'opencv-contrib-python-headless==4.8.1.78',
        'pandas==2.1.1',
        'scikit-learn==1.3.1',
        'natsort==8.4.0',
        'glob2==0.7',
        'shapely == 1.7',
        'imageio==2.33',
        'Pillow==10.1.0',
                      ],

    classifiers=[
        'Development Status :: 1 - Review',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)