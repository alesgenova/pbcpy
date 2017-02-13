LEN_UNITS = ['Bohr', 'Angstrom', 'nm', 'm']

LEN_CONV = {}
LEN_CONV['Bohr'] = {'Bohr': 1.0, 'Angstrom': 0.5291772106712,
                    'nm': 0.05291772106712, 'm': 5.291772106712e-11}
LEN_CONV['Angstrom'] = {'Bohr': 1.8897261254535427,
                        'Angstrom': 1.0, 'nm': 1.0e-1, 'm': 1.0e-10}
LEN_CONV['nm'] = {'Bohr': 18.897261254535427,
                  'Angstrom': 10., 'nm': 1.0, 'm': 1.0e-9}
LEN_CONV['m'] = {'Bohr': 1.8897261254535427e10,
                 'Angstrom': 1.0e10, 'nm': 1.0e9, 'm': 1.0}
