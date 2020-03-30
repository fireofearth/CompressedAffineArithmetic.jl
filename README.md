# AffineArithmetic.jl

Julia implementation of affine arithmetic based on the C++ library [aaflib](#related-software). Uses sparse vector storage of coefficients, and Chebyshev approximation for non-affine operations.

## Dependencies

[IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)  

## TODOs

- Allow affine forms to be instantiated with coefficients of any type `<: Real`.

## Related Works

Jorge Stolfi and Luiz Henrique de Figueiredo. 1997. Self-Validated Numberical Methods and Applications. In *Proceedings of the 21st Brazilian Mathematics Colloquium*. ??? PDF:<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.8089&rep=rep1&type=pdf>

## Related Software

[AffineArithmetic.jl](https://github.com/JuliaIntervals/AffineArithmetic.jl): an alternative implementation.

Affine Arithmetic C++ Library (aaflib) 
<http://aaflib.sourceforge.net>






