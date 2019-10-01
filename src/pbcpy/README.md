# Pythran

## Use the pythran to accelerate the code

### Normal build

pythran -march=native -DUSE_BOOST_SIMD -Ofast math_thran.py

### With OpenMP

pythran -march=native -fopenmp -DUSE_XSIMD -Ofast math_thran.py

## Tips

   For the numpy which build with intel-MKL, it useless.
