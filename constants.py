
len_units = ['Bohr', 'Angstrom', 'nm', 'm']
len_conv = {}

len_conv['Bohr'] = {'Bohr': 1.0, 'Angstrom': 0.5291772106712,
                    'nm': 0.05291772106712, 'm': 5.291772106712e-11}
len_conv['Angstrom'] = {'Bohr': 1.8897261254535427,
                        'Angstrom': 1.0, 'nm': 1.0e-1, 'm': 1.0e-10}
len_conv['nm'] = {'Bohr': 18.897261254535427,
                  'Angstrom': 10., 'nm': 1.0, 'm': 1.0e-9}
len_conv['m'] = {'Bohr': 1.8897261254535427e10,
                 'Angstrom': 1.0e10, 'nm': 1.0e9, 'm': 1.0}
