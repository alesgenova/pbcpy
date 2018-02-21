from setuptools import setup, find_packages
import re
import sys
SRC_DIR = "./src"
if SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)
from pbcpy import __version__, __author__, __contact__, __license__

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except (IOError, ImportError):
    long_description = open('README.md').read()
    long_description_old = '''
pbcpy is a Python3 package providing some useful tools when dealing with
molecules and materials under periodic boundary conditions (PBC).
Foundation of the package are the Cell and Coord classes, which define a unit cell under PBC, and a cartesian/crystal coordinate respectively.
pbcpy also provides tools to deal with quantities represented on an equally spaced grids, through the Grid and Field classes. Operations such as interpolations or taking arbitrary 1D/2D/3D cuts are made very easy.
In addition, pbcpy exposes a fully periodic N-rank array, the pbcarray, which is derived from the numpy.ndarray.
Finally, pbcpy provides IO support to some common file formats:
The Quantum Espresso .pp format (read only)
The XCrySDen .xsf format (write only) 
   '''



setup(
    name='pbcpy',
    description='''A toolbox to make it easier to deal with materials under periodc boundary conditions.''',
    long_description=long_description,
    version=__version__,
    url='https://gitlab.com/ales.genova/pbcpy/',
    author=__author__,
    author_email=__contact__,
    license=__license__,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    packages=find_packages('src'),  # include all packages under src
    package_dir={'':'src'},   # tell distutils packages are under src
    install_requires=[
        'numpy>=1.6.0',
        'scipy>=0.10.0'
    ]
)
