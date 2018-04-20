from setuptools import setup, find_packages
import re
import sys
import os
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./src")
if SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)
from pbcpy import __version__, __author__, __contact__, __license__

readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()
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
    description="A toolbox for easy handling of materials under periodc boundary conditions.",
    long_description=readme,
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
    include_package_data = True,
    install_requires=[
        'numpy>=1.6.0',
        'scipy>=0.10.0'
    ]
)
