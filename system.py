'''
    pbcpy is a python package to seamlessly tackle periodic boundary conditions.

    Copyright (C) 2016 Alessandro Genova (ales.genova@gmail.com).

    pbcpy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    pbcpy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with pbcpy.  If not, see <http://www.gnu.org/licenses/>.
'''


class System(object):

    def __init__(self, ions, cell, name=None, plot=None):

        self.ions = ions
        self.cell = cell
        self.name = name
        self.plot = plot
