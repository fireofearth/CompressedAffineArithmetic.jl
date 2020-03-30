module CompressedAffineArithmetic 
#=
 # Compressed Affine arithmetic module
 #
 # This is a implementation of the An Affine Arithmetic C++
 # Library ( http://aaflib.sourceforge.net ) in Julia
 #
 # Specification:
 # - we use Float64 to represent double, and Vector{Float64}
 # to represent arrays in heap memory.
 # - Julia operations rely on the use of the `last` index to
 # keep track of indeterminate coefficients
 #
 # TODO: finish documentation
 # TODO: complete + test support for ForwardDiff
 # TODO: enable iterator
 # TODO: make `Affine <: Real` into `Affine{T<:Real} <: Number`; do package type support
 # the Julian way; support for T<:Real coefficients
 #
 # TODO: aaflib trig. functions sin, cos; requires me to know
 # what implementation changes Goubault/Putot made for sin, cos
 # TODO: figure out what changes Goubault/Putot made for aaflib
=#

import IntervalArithmetic: Interval, interval, mid, radius

# since we will likely use AffineArithmetic along with IntervalArithmetic, we want to avoid namespace conflicts so we will import inf, sup here
import IntervalArithmetic: inf, sup


import Base:
    zero, one, iszero, isone, convert, isapprox, promote_rule,
    isnan, isinf, isfinite,
    getindex, length, show, size, firstindex, lastindex,
    min, max, conj,
    <, <=, >, >=, ==, +, -, *, /, inv, ^, sin, cos, abs2,
    issubset, ⊆, ⊇

using Logging

export
    zero, one, iszero, isone, convert, isapprox, promote_rule,
    isnan, isinf, isfinite,
    getindex, length, show, size, firstindex, lastindex,
    <, <=, >, >=, ==, +, -, *, /, inv, ^, sin, cos, abs2, 
    compact,
    AffineCoeff, AffineInd, AffineInt, Affine, affineTOL,
    Interval, interval, inf, sup,
    rad, min, max, conj, absmin, absmax,
    issubset, ⊆, ⊇

# Used in test settings.
export getLastAffineIndex, resetLastAffineIndex, ApproximationType

 #=
 # Module-wide constants
 #
 # TODO: Can we simplify the usage of constants?
=#
@enum ApproximationType MINRANGE CHEBYSHEV SECANT
MINRAD  = 1E-10
TOL     = 1E-15
EPSILON = 1E-20
NOISE   = 1E-5

# tolerance to export
affineTOL = TOL

 #=
 # Type declarations
=#
Float       = Float64
AffineCoeff = Float64 # type for coefficients and constants
AffineInt   = Int64 # type for integers
AffineInd   = Int64 # type for indexes
approximationType = CHEBYSHEV # default rounding mode

disp(msg) = print("$(msg)\n")
debug() = print("DEBUG\n")

 #=
 # lastAffineIndex keeps record of last coefficient index of affine forms accoss all Affine 
 # instances. lastAffineIndex is used to assign indexes to new deviation symbols.
 #
 # Specification:
 # - force setLastAffineIndex to call whenever a new Affine instance is created.
 #
 # TODO: test methods
 # TODO: turn this into a decorator/macro and force calls?
=#
let lastAffineIndex::Int = 0

    global function resetLastAffineIndex()
        lastAffineIndex = 0
    end

    global function getLastAffineIndex()
        return lastAffineIndex
    end

    global function makeAffineIndex()
        lastAffineIndex += 1
        return lastAffineIndex
    end

     #=
     # Assign new coefficient index i for noise symbol μᵢ.
     # To be used when assigning computation noise.
     # Similar to inclast() in aaflib
    =#
    global function addAffineIndex()
        lastAffineIndex += 1
        return [lastAffineIndex]
    end

    global function addAffineIndex(indexes::Vector{Int})
        lastAffineIndex += 1
        return vcat(indexes, lastAffineIndex)
    end

     #=
     # Given an array representing indexes of affine form, set lastAffineIndex to the last
     # index only if the last index is larger than lastAffineIndex
    =#
    global function setLastAffineIndex(indexes::Vector{Int})
        if(!isempty(indexes))
            m = last(indexes)
            if(m > lastAffineIndex)
                lastAffineIndex = m
            end
        end
        return indexes
    end
end

 #=
 # Affine represents an affine form
 #
 # Specification:
 # - Affine is a collection with deviations as its elements.
 #
 # Invariants:
 # - Affine indexes are always in sorted order from lowest to highest
 # - elts in Affine indexes are unique
=#
struct Affine{T <: Real} <: Number
    center::T  # central value 
    deviations::Vector{T}
    indexes::Vector{Int}

     #=
     # Creates an Affine with deviations
    =#
    function Affine(v0::T, dev::Vector{T}, ind::Vector{Int})
        @assert length(ind) == length(dev)
        for ii in 1:(length(ind) - 1)
            @assert ind[ii] < ind[ii + 1]
        end
        new(v0, dev, setLastAffineIndex(ind))
    end  
end

 #=
 # Affine constructors for initialization
=#
Affine(a::Affine) = a
Affine(X::Interval{T}) where T <: Real = Affine(T(mid(X)), [T(radius(X))], addAffineIndex())
Affine(x::T) where T <: Real = Affine(x, Vector{T}(), Vector{Int}())

function Affine(x::T, v::Vector{T}) where T <: Real
    idx = Vector{T}(UndefInitializer(), length(v))
    for i = 1:length(v)
        idx[i] = makeAffineIndex()
    end
    return Affine(x, v, idx)
end

function Affine(l::T, h::T) where T <: Real
    @assert l ≤ h   
    if(l == h)
        return Affine(l)
    else
        return Affine(T((l + h) /2), [T((h - l) / 2)], addAffineIndex())
    end
end

 #=
 # Constructor that assigns new center to Affine
 # used in affine operations
=#
Affine(a::Affine{S}, cst::T) where S <: T <: Real = 
        Affine(cst, Vector{S}(a.deviations), a.indexes)
Affine(a::Affine{S}, cst::T) where T <: S <: Real = 
        Affine(S(cst), a.deviations, a.indexes)

 #=
 # Constructor that assigns new center, and diff to Affine. We assume indexes unchanged
 # used in affine operations
=#
Affine(a::Affine{S}, cst::AffineCoeff{T}, dev::Vector{T}) where {S, T} <: Real = 
        Affine(cst, dev, a.indexes)

function getindex(a::Affine{T}, ind::Int)::T where T <: Real
    if(ind < 0 || ind > length(a.deviations))
        throw(BoundsError(a.deviations, ind))
    elseif(ind == 0)
        return a.center
    else
        return a.deviations[ind]
    end
end

length(a::Affine) = length(a.deviations)

 #=
 # Write representation of affine form to current output stream
=#
function show(io::IO, a::Affine)
    s = "$(a[0])"
    if(length(a) > 0)
        for i in 1:length(a)
            s *= " + $(a[i])ϵ$(a.indexes[i])"
        end
    end
    print(io::IO, s)
end

 #=
 # Obtain the string representation of an affine form
 # TODO: TESTING ONLY delete when done.
=#
function reprit(center, deviation, indexes)
    s = "$(center)"
    if(length(deviation) > 0)
        for i in 1:length(deviation)
            s *= " + $(deviation[i])ϵ$(indexes[i])"
        end
    end
    return s
end

 #=
 # convert, one, zero, isone, iszero
 # is already supported by Base module, provided Affine <: Real or Affine <: Number
=#
convert(::Type{Affine}, x::Affine) = Affine(x)
convert(::Type{Affine}, x::T) where T <: Real = Affine(x)

#one(::Type{Affine})  = convert(Affine, 1.)
#one(x::Affine)       = convert(Affine, 1.)
#zero(::Type{Affine}) = convert(Affine, 0.)
#zero(x::Affine)      = convert(Affine, 0.)

#isone(x::Affine)     = x == one(Affine)
#iszero(x::Affine)    = x == zero(Affine)

isnan(x::Affine) = any(isnan.(x.deviations))
isinf(x::Affine) = any(isinf.(x.deviations))
isfinite(x::Affine) = !isnan(x) && !isinf(x)

promote_rule(::Type{Affine}, ::Type{Affine}) = Affine
promote_rule(::Type{Affine}, ::Type{T}) where T <: Real = Affine
promote_rule(::Type{T}, ::Type{Affine}) where T <: Real = Affine

 #=
 # Get the total deviation of an Affine (i.e. the sum of all deviations (their abs value))
=#
rad(a::Affine) = sum(abs.(a.deviations))

  #=
  # Getters of the bounds of an Affine
 =#
max(a::Affine)::AffineCoeff = a[0] + rad(a)
min(a::Affine)::AffineCoeff = a[0] - rad(a)
absmax(a::Affine)::AffineCoeff = rad(a) |> x -> max(abs(a[0] - x), abs(a[0] + x))
absmin(a::Affine)::AffineCoeff = rad(a) |> x -> min(abs(a[0] - x), abs(a[0] + x))

Interval(a::Affine) = Interval(min(a), max(a))
interval(a::Affine) = Interval(a)

firstindex(a::Affine) = length(a) > 0 ? a.indexes[1] : 0
lastindex(a::Affine)  = length(a) > 0 ? last(a.indexes) : 0 

  #=
  # Conditionals
 =#
<(a::Affine,  p::Affine) = (a[0] + rad(a)) <  (p[0] - rad(p))
<=(a::Affine, p::Affine) = (a[0] + rad(a)) <= (p[0] - rad(p))
>(a::Affine,  p::Affine) = (a[0] - rad(a)) >  (p[0] + rad(p))
>=(a::Affine, p::Affine) = (a[0] - rad(a)) >= (p[0] + rad(p))
issubset(a::Affine,  p::Affine) = Interval(a) ⊆ Interval(p)

 #=
 # Equality
 # Specification: affine equality compares cvalues, and deviations up to
 # some tolerance which defaults to TOL
 #
 # TODO: simplify?
=#
function equalityInternal(a::Affine, p::Affine; tol::Float64=TOL)
    if(length(a) != length(p))
        return false
    end

    if(abs(a[0]) < 1 && abs(p[0]) < 1)
        if(abs(a[0] - p[0]) > tol)
            return false
        end
    else
        if(abs((a[0] - p[0]) / (a[0] + p[0])) > tol)
            return false
        end
    end

    for i in 1:length(a)
        if(a.indexes[i] != p.indexes[i])
            return false
        end
        if(abs(a[i]) < 1 && abs(p[i]) < 1)
            if(abs(a[i] - p[i]) > tol)
                return false
            end
        else
            if(abs(a[i] - p[i]) / 
                   (abs(a[i]) + abs(p[i])) > tol)
                return false
            end
        end
    end
    
    return true
end

==(a::Affine, p::Affine) = equalityInternal(a, p)

 #=
 # true iff two affine forms are approximate
 #
 # TODO: may this function properly extend Base.isapprox
=#
function isapprox(a::Affine, p::Affine; tol::Float64=TOL)
    return equalityInternal(a, p; tol=tol)
end

conj(a::Affine) = a

+(a::Affine, cst::AffineCoeff)::Affine = Affine(a, a[0] + cst)
-(a::Affine, cst::AffineCoeff)::Affine = Affine(a, a[0] - cst)
+(cst::AffineCoeff, a::Affine)::Affine = Affine(a, cst + a[0])
-(cst::AffineCoeff, a::Affine)::Affine = Affine(a, cst - a[0])

+(a::Affine, cst::AffineInt)::Affine = Affine(a, convert(AffineCoeff, a[0] + cst))
-(a::Affine, cst::AffineInt)::Affine = Affine(a, convert(AffineCoeff, a[0] - cst))
+(cst::AffineInt, a::Affine)::Affine = Affine(a, convert(AffineCoeff, cst + a[0]))
-(cst::AffineInt, a::Affine)::Affine = Affine(a, convert(AffineCoeff, cst - a[0]))

*(a::Affine, cst::AffineCoeff)::Affine = Affine(a, a[0] * cst, cst * a.deviations)
*(cst::AffineCoeff, a::Affine)::Affine = Affine(a, cst * a[0], cst * a.deviations)

*(a::Affine, cst::AffineInt)::Affine = Affine(a, convert(AffineCoeff, a[0] * cst), 
    convert(Vector{AffineCoeff}, cst * a.deviations))
*(cst::AffineInt, a::Affine)::Affine = Affine(a, convert(AffineCoeff, cst * a[0]), 
    convert(Vector{AffineCoeff}, cst * a.deviations))

 #=
 # a + p, where a, p are Affine
=#
function +(a::Affine, p::Affine)::Affine
    if(length(p) == 0)
        return a + p[0]
    elseif(length(a) == 0)
        return a[0] + p
    end
    indt = [ii for ii in union(a.indexes, p.indexes)]
    sort!(indt)
    devt = fill(0.0, length(indt))
    pcomp = tuple.(indexin(indt, a.indexes), indexin(indt, p.indexes))
    for (ii, (ia,ip)) in enumerate(pcomp)
        @assert ia != nothing || ip != nothing
        devt[ii] = (ia == nothing ? p[ip] : 
         (ip == nothing ? a[ia] : a[ia] + p[ip]))
    end
    return Affine(a[0] + p[0], devt, indt)
end

 #=
 # a - p where a, p are Affine
=#
function -(a::Affine, p::Affine)::Affine
    if(length(p) == 0)
        return a - p[0]
    elseif(length(a) == 0)
        return a[0] - p
    end
    indt = [ii for ii in union(a.indexes, p.indexes)]
    sort!(indt)
    devt = fill(0.0, length(indt))
    pcomp = tuple.(indexin(indt, a.indexes), indexin(indt, p.indexes))
    for (ii, (ia,ip)) in enumerate(pcomp)
        @assert ia != nothing || ip != nothing
        devt[ii] = (ia == nothing ? -p[ip] : 
         (ip == nothing ? a[ia] : a[ia] - p[ip]))
    end
    return Affine(a[0] - p[0], devt, indt)
end


+(a::Affine) = Affine(a)
-(a::Affine) = Affine(a, -a[0], -a.deviations)

 #=
 # All non-affine operations (*,/,inv,^,sin,cos) default to Chebyshev approximation.
 #
 # Specification for univariate Chebyshev approximation of bounded, twice differentiable
 # f : R -> R and affine form x = x₀ + ∑ᴺᵢxᵢϵᵢ
 #
 # 1. let a = x₀ - ∑ᴺᵢ|xᵢ|, and b = x₀ + ∑ᴺᵢ|xᵢ|. We require f''(u) ≠ 0 for u ∈ (a, b)
 # 2. let α = (f(b) - f(a)) / (b - a) be the slope of the line l(x) that interpolates the
 # points (a, f(a)) and (b, f(b)). l(x) = αx + (f(a) - αa)
 # 3. solve for u ∈ (a, b) such that f'(u) = α. By Mean-value theorem u must exists.
 # 4. ζ = ½(f(u) + l(u)) - αu
 # 5. δ = ½|f(u) - l(u)|
 #
 # Specification for bivariate Chebyshev approximation of ??? f : R² -> R and affine
 #
=#

 #=
 # Approximates a * p where a, p are Affine
 # There are three affine products based on the appoximation of the coefficient for μₖ
 #   xy = x₀ŷ₀ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + ½∑[over 1⩽i,j⩽n] |xᵢyⱼ+yᵢxⱼ|μₖ
 #   xy = x₀ŷ₀ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + (∑ᴺᵢ|xᵢ|)(∑ᴺᵢ|yᵢ|)μₖ
 #   xy = x₀ŷ₀ + ½∑ᴺᵢxᵢyᵢ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + [(∑ᴺᵢ|xᵢ|)(∑ᴺᵢ|yᵢ|) - ½∑ᴺᵢ|xᵢyᵢ|]μₖ
 # The last approximation is obtainable by observing that for products of like terms 
 # we have squares of noise symbols: xᵢyᵢϵᵢ² 
 # Since ϵᵢ² ∈ [0,1] the term has a center at ½xᵢyᵢ with magnitude of deviation ½|xᵢyᵢ|
 #
 # Specification:
 #   xy = x₀ŷ₀ + ½∑ᴺᵢxᵢyᵢ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + [(∑ᴺᵢ|xᵢ|)(∑ᴺᵢ|yᵢ|) - ½∑ᴺᵢ|xᵢyᵢ|]μₖ
=#
function *(a::Affine, p::Affine)::Affine
    if(length(p) == 0)
        return a * p[0]
    elseif(length(a) == 0)
        return a[0] * p
    end

    # create new index with length = length(a.indexes) + length(p.indexes) + 1
    adjDeviation2 = 0.0
    adjCenter2    = 0.0
    indt  = [ii for ii in union(a.indexes, p.indexes)]
    sort!(indt)
    indt  = addAffineIndex(indt)
    lindt = length(indt)
    devt  = fill(0.0, lindt)
    pcomp = tuple.(indexin(indt, a.indexes), indexin(indt, p.indexes))
    for (ii, (ia,ip)) in enumerate(pcomp)
        if(ia == nothing && ip == nothing) # happens at the end of indt
            # do nothing
        elseif(ia == nothing)
            devt[ii] = a[0] * p[ip]
        elseif(ip == nothing)
            devt[ii] = a[ia] * p[0]
        else
            devt[ii] = a[ia] * p[0] + a[0] * p[ip]
            adjDeviation2 += abs(a[ia] * p[ip])
            adjCenter2    += a[ia] * p[ip]
        end
    end
    devt[lindt] = rad(a)*rad(p) - 0.5*adjDeviation2
    #@info "in function *(a::Affine, p::Affine)\na = $(repr(a))\np = $(repr(p))\nout = $(reprit(a[0]*p[0] + 0.5*adjCenter2, devt, indt))"
    return Affine(a[0]*p[0] + 0.5*adjCenter2, devt, indt)
end

 #=
Obtain 1/a where a is Affine

Specification:
- Inverse does not exist if 0 ∈ [p[0] - rad(p), p[0] + rad(p)] (which is given by guard 
a*b < EPSILON)
- uses Chebyshev approximation:

1. let a = x₀ - ∑ᴺᵢ|xᵢ|, and b = x₀ + ∑ᴺᵢ|xᵢ|.

2. let α = -1/ab be the slope of the interpolation line l(x)

3. solve for u ∈ (a, b) such that -1/u² = -1/ab
If a, b > 0, then u = √ab and l(u) = 1/a + 1/b - 1/√ab
If a, b < 0, then u = -√ab and l(u) = 1/a + 1/b + 1/√ab

4.
If a, b > 0, then ζ = ½(1/a + 1/b) + 1/√ab
If a, b < 0, then ζ = ½(1/a + 1/b) - 1/√ab

5. δ = ½|f(u) - l(u)|
If a, b > 0, then δ = ½(1/a + 1/b) - 1/√ab
If a, b < 0, then δ = -½(1/a + 1/b) - 1/√ab

# TODO: clean code, refactor
=#
function inv(p::Affine)::Affine
    if(length(p) == 0)
        return Affine(1.0 / p[0])
    end

    r = rad(p)
    a = p[0] - r;
    b = p[0] + r;
    if(a*b < EPSILON) # if 0 ∈ [p[0] - rad(p), p[0] + rad(p)]
        throw(DomainError(p, "trying to invert zero"))
    end
    inva = 1.0 / a
    invb = 1.0 / b

    if(approximationType == CHEBYSHEV)
        alpha = -inva * invb
        u = sqrt(a * b)

        if(a > 0) # affine is above 0
            delta = 0.5*(inva + invb - 2.0/u)
            dzeta = inva + invb - delta
        else # affine is below 0
            delta = -0.5*(inva + invb + 2.0/u)
            dzeta = inva + invb + delta
        end
        
    elseif(approximationType == MINRANGE)
        error("inv: approx. method incomplete")
    else # if(approximationType == SECANT)
        error("inv: approx. method incomplete")
    end

    indt = addAffineIndex(p.indexes)
    devt = alpha * p.deviations
    devt = vcat(devt, delta)
    return Affine(alpha*p[0] + dzeta, devt, indt)
end

function /(a::Affine, cst::AffineCoeff)::Affine
    if(cst == zero(cst))
        throw(DomainError(a, "trying to divide by zero"))
    end
    return Affine(a, a.cvalue * (1.0 / cst), (1.0 / cst) * a.deviations)
end

function /(a::Affine, cst::AffineInt)::Affine
    if(cst == zero(cst))
        throw(DomainError(a, "trying to divide by zero"))
    end
    return Affine(a, a.cvalue * (1.0 / cst), (1.0 / cst) * a.deviations)
end

/(cst::AffineCoeff, a::Affine)::Affine = cst * inv(a)
/(cst::AffineInt,   a::Affine)::Affine = cst * inv(a)
/(a::Affine, p::Affine)::Affine = a * inv(p)

 #=
Obtain a^n where a is Affine and n is an integer

1. let a = x₀ - ∑ᴺᵢ|xᵢ|, and b = x₀ + ∑ᴺᵢ|xᵢ|

2. let α = (b^n - a^n) / (b - a) be the slope of the line l(x)

3. solve for u ∈ (a, b) such that n*u^(n - 1) = α
u = ± |α / n|^(1/(n - 1))


4. ζ = ½(f(u) + l(u)) - αu
5. δ = ½|f(u) - l(u)|

=#
function ^(p::Affine, n::Int)
    if(length(p) == 0)
        return Affine(p[0]^n)
    end

    if(n == 0)
        return one(Affine)
    elseif(n == 1)
        return p
    elseif(n == -1)
        return inv(p)
    end
    
    r = rad(p)
    a = p[0] - r
    b = p[0] + r
    powa = a^n
    powb = b^n
    if(a*b < EPSILON && n < 0)
        throw(DomainError(p, "trying to invert zero"))
    end

    if(approximationType == CHEBYSHEV)
        if(r > MINRAD)
            alpha = (powb - powa) / (b - a)
        else
            alpha = n * powa / (a + EPSILON)
        end

        xₐ = -abs(alpha / n)^(1.0 / (n - 1.0))
        xᵦ = -xₐ

        if(xₐ > a)
            powxₐ = xₐ^n
        else
            xₐ  = a
            powxₐ = powa
        end

        if(xᵦ < b)
            powxᵦ = xᵦ^n
        else
            xᵦ  = b
            powxᵦ = powb
        end

        yₐ= powxₐ - alpha*xₐ
        yᵦ= powxᵦ - alpha*xᵦ
        delta = 0.5*(yₐ - yᵦ)
        dzeta = 0.5*(yₐ + yᵦ)

    elseif(approximationType == MINRANGE)
        error("incomplete")
    else # if(approximationType == SECANT)
        error("incomplete")
    end

    indt = addAffineIndex(p.indexes)
    devt = alpha * p.deviations
    devt = vcat(devt, delta)
    return Affine(alpha*p[0] + dzeta, devt, indt)
end

abs2(p::Affine) = p^2

 #=
 # obtain sin(p) where p is Affine
 #
 # TODO: documentation
 # TODO: should we compact p?
=#
function sin(p::Affine)::Affine
    if(length(p) == 0)
        return Affine(sin(p[0]))
    end
   
    r = rad(p)
    a = p[0] - r
    b = p[0] + r
    if(a > π || a < -π)
        a -= 2*π * floor(a / (2*π))
        a -= (a > π) ? 2*π : 0.0
        b = a + 2*r
    end
    sina = sin(a)
    sinb = sin(b)

    if(b < 0 || (b < π && a > 0))
        # Use Chebyshev approximation
        if(r > 1.0E-6)  
            alpha = (sinb - sina) / (b - a)
            sol   = sign(a)*acos(alpha)
            # Here sin ∘ acos(x) = √(1 - x²)
            fsol  = √(1 - alpha^2)
            dzeta = (sina + fsol - alpha*(a + sol)) / 2
            delta = abs(fsol - sina - alpha*(sol - a)) / 2
        else
            alpha = cos(a)
            dzeta = sina - alpha*a
            delta = 0.0
        end
    else
        # Use min range optimization
        if(a <= 0)
            alpha = 1
            delta = (-sinb + sina - alpha*(a - b)) / 2
        else
            alpha = -1
            delta = (sinb - sina + alpha*(a - b)) / 2
        end
        dzeta = (sina + sinb - alpha*(a + b)) / 2
    end

    indt = addAffineIndex(p.indexes)
    devt = alpha * p.deviations
    devt = vcat(devt, delta)
    return Affine(alpha*p[0] + dzeta, devt, indt)
end

 #=
 # obtain cos(p) where p is Affine
=#
function cos(p::Affine)::Affine
    return sin(p + π/2.0)
end

 #=
 # Removes all noise terms xᵢϵᵢ coefficients with |xᵢ| < tol
=#
function compact(p::Affine; tol::Float64=TOL)
    pfilt = filter(x -> abs(x[1]) > TOL, tuple.(p.deviations, p.indexes))
    devt = [x[1] for x in pfilt]
    indt = [x[2] for x in pfilt]
    return Affine(p[0], devt, indt)
end

compact(x::Vector{Affine}; tol::Float64=TOL) = (p -> compact(p, tol=tol)).(x)
compact(x::Matrix{Affine}; tol::Float64=TOL) = (p -> compact(p, tol=tol)).(x)

 #=
 # Sum all noise coefficients that are below a threshold and add them to a new noise term
 #
 #
 # TODO finish this
=#
#function aggregate(p::Affine, level::Float64=TOL_NOISE)
#    
#    devThreshold = abs(level * rad(p))
#    if(devThreshold < TOL)
#        devThreshold = TOL
#    end
#
#    for()
#    end
#end

end # module AffineArithmetic