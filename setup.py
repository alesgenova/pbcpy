from setuptools import setup, find_packages

setup(name='pbcpy',
      description='''A toolbox to make it easier to deal with materials under
                    periodc boundary conditions.''',
      long_description='''pbcpy is a Python3 package providing some useful tools when dealing with
                          molecules and materials under periodic boundary conditions (PBC).
                          Foundation of the package are the Cell and Coord classes, which define a unit cell under PBC, and a cartesian/crystal coordinate respectively.
                          pbcpy also provides tools to deal with quantities represented on an equally spaced grids, through the Grid and Plot classes. Operations such as interpolations or taking arbitrary 1D/2D/3D cuts are made very easy.
                          In addition, pbcpy exposes a fully periodic N-rank array, the pbcarray, which is derived from the numpy.ndarray.
                          Finally, pbcpy provides IO support to some common file formats:
                            The Quantum Espresso .pp format (read only)
                            The XCrySDen .xsf format (write only) ''',
      version='17.02',
      url='https://gitlab.com/ales.genova/pbcpy/',
      author='A. Genova',
      author_email='ales.genova@gmail.com',
      license='MIT',
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
