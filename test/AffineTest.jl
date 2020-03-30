using Test, Random, Dates
using IntervalArithmetic
using CompressedAffineArithmetic
using Logging

 #=
 # Affine arithmetic testing
 # 
 # Specifications:
 # - algebriac identities one, zero, isone, iszero
 # - unitary operations and constructors
 # - detailed constructors
 #
 # - when comparing two affine forms we require the indexes to match thus we call 
 # resetLastAffineIndex() before generating them. This is especially true when testing for
 # ForwardDiff
 #
 # TODO: improve tests on constructors, getters, utility functionality
 # TODO: add more (~4) cases to compare solutions with aaflib
 # TODO: improve random generators of test cases
 # TODO: tests to check compatibility with ForwardDiff
 # TODO: add test to verify that affine forms under functions do act as bounds for outputs
 # of those functions
 #
 # Remark: it is not possible to recreate which indexes are assigned to noise symbols when
 # using ForwardDiff. Now using `sameForm()` instead of `==`, and compact()
=#

Random.seed!(Dates.value(Dates.now()))
MIN = -10^4
MAX =  10^4
U   = -1 .. 1

 #=
 # Comparable to `==` except we do not check indexes. Used for testing only.
=#
function sameForm(a::Affine, p::Affine; tol::Float64=affineTOL)
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

sameForm(x::Vector{Affine}, y::Vector{Affine}; tol::Float64=affineTOL) = sameForm.(x, y, tol=tol) |> x -> reduce((b1, b2) -> b1 && b2, x)
sameForm(X::Matrix{Affine}, Y::Matrix{Affine}; tol::Float64=affineTOL) = sameForm.(X, Y, tol=tol) |> x -> reduce(&, x, dims=1) |> x -> reduce(&, x)

 #=
 # Affine Arithmetic Common
 # All functionality except for elementary functions and binary operations
=#
@testset "affine arithmetic common" begin
    xl = rand(MIN .. MAX)
    xh = rand(xl .. MAX)
    vlen = 10^3
    v = [rand(U) for x in 1:vlen]
    x  = rand(MIN .. MAX)
    
    @testset "convert" begin
        center = 3.14
        a = convert(Affine, center)
        @test a == Affine(center)
        @test a[0] == center
        @test length(a) == 0
    end

    @testset "algebriac identities" begin
        center = 3.14
        dev    = [0.75, 0.01]
        ind    = [1, 3]
        a = Affine(center, dev, ind)
        aOne = one(a)
        aZero = zero(a)
        @test a*aOne == a == aOne*a
        @test a + aZero == a == aZero + a
    end

    @testset "constructors; unitary" begin
        center1 = 100.001
        dev1    = [2.0, 1.0, -5.0, 4.0]
        ind1    = [1  , 3  ,  4,   6]
        a1 = Affine(center1, dev1, ind1)
        @test a1[0] == center1
        for ii in 1:4
            @test a1[ii] == dev1[ii]
            @test a1.indexes[ii] == ind1[ii]
        end
        @test rad(a1) == 2.0 + 1.0 + 5.0 + 4.0
        @test length(a1) == 4
        @test Interval(a1) == Interval(a1[0] - rad(a1), a1[0] + rad(a1))
    end

    @testset "range constructors" begin
        a = Affine(xl, xh)
        @test a[0] ≈ (xl + xh) / 2
        @test a[1] ≈ (xh - xl) / 2
        @test length(a) == 1
        a = Affine(xl .. xh)
        @test a[0] ≈ (xl + xh) / 2
        @test a[1] ≈ (xh - xl) / 2
        @test length(a) == 1
    end

    @testset "array constructor" begin
        resetLastAffineIndex()
        Affine(xl, xh) * Affine(xl, xh)
        a = Affine(x, v)
        @test a[0] == x
        @test length(a) == vlen
        for i in 1:vlen
            @test a[i] == v[i]
            @test a.indexes[i] == i+3
        end
    end

    @testset "set comparison" begin
        a1 = Affine(rand(xl .. xh))
        a2 = Affine(xl, xh)
        a3 = Affine(xl - eps(), xh + eps())
        @test a1 ⊆ a2 ⊆ a3
        a1 = Affine(x, v)
        a2 = Affine(x, [v; rand(U)])
        @test a1 ⊆ a2
        a2 = Affine(x, sort(v))
        @test a1 ⊆ a2
        @test a1 ⊇ a2
    end

    @testset "equality" begin
        center1 = 1.321
        dev1    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind1    = [1  , 2,   5  ,  7,   8]
        a1 = Affine(center1, dev1, ind1)
        center2 = 1.321
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 2,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)

        @test a1 == a2
        ind1[2] = 3
        a1 = Affine(center1, dev1, ind1)
        @test a1 != a2
        ind1[2] = 2
        a1 = Affine(center1, dev1, ind1)
        @test a1 == a2
        dev1[3] = 0.5
        a1 = Affine(center1, dev1, ind1)
        @test a1 != a2
        dev1[3] = 1.5
        a1 = Affine(center1, dev1, ind1)
        @test a1 == a2
    end

    @testset "equality boundary conds" begin
        center1 = 1.321
        dev1    = [2.1, 0.0, 1.5, -5.3, 4.0]
        ind1    = [1  , 2,   5  ,  7,   8]
        a1 = Affine(center1, dev1, ind1)
        center2 = 1.321
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 2,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)

        @test a1 != a2
        dev1    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind1    = [1  , 2,   5  ,  7,   9]
        a1 = Affine(center1, dev1, ind1)
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 2,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)
        @test a1 != a2
        dev1    = [1.1, 0.0, 1.5, -5.3, 4.2]
        ind1    = [1  , 2,   5  ,  7,   8]
        a1 = Affine(center1, dev1, ind1)
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 2,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)
        @test a1 != a2
        dev1    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind1    = [2  , 3,   5  ,  7,   8]
        a1 = Affine(center1, dev1, ind1)
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 3,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)
        @test a1 != a2
        center1 = 0.321
        dev1    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind1    = [1  , 2,   5  ,  7,   8]
        a1 = Affine(center1, dev1, ind1)
        center2 = 1.321
        dev2    = [2.1, 0.0, 1.5, -5.3, 4.2]
        ind2    = [1  , 2,   5  ,  7,   8]
        a2 = Affine(center2, dev2, ind2)
        @test a1 != a2
    end

    @testset "compact" begin
        center = 12.0
        dev = [0.0, 2.0, 3.0, 0.0, 1.0, 0.0]
        ind = [1,   2,   3,   5,   8,   9]
        a = Affine(center, dev, ind)
        @test compact(a) == Affine(center, [2.0, 3.0, 1.0], [2, 3, 8])
        a = Affine(center, [0.0], [1])
        @test compact(a) == Affine(center)
        a = Affine(center)
        @test compact(a) == Affine(center)
    end
end

 #=
 # Verify Affine-Constant operations where constant can be floats or ints.
=#
@testset "affine constant arithmetic ops" begin
    center = 12.0
    dev    = [3.0, 6.0, 9.0]
    ind    = [1,    3,    4]
    a = Affine(center, dev, ind)

    @testset "neg" begin
        @test -a == Affine(-center, -dev, ind)
    end

    @testset "addition / subtraction" begin
        @test a + 2.0 == Affine(14.0, dev, ind)
        @test a + 2   == Affine(14.0, dev, ind)
        @test 2.0 + a == Affine(14.0, dev, ind)
        @test 2   + a == Affine(14.0, dev, ind)
        @test a - 2.0 == Affine(10.0, dev, ind)
        @test a - 2   == Affine(10.0, dev, ind)
        @test 2.0 - a == Affine(-10.0, dev, ind)
        @test 2   - a == Affine(-10.0, dev, ind)
    end

    @testset "multiplication" begin
        @test a * 2.0 == Affine(24.0, [6.0, 12.0, 18.0], ind)
        @test a * 2   == Affine(24.0, [6.0, 12.0, 18.0], ind)
        @test 2.0 * a == Affine(24.0, [6.0, 12.0, 18.0], ind)
        @test 2   * a == Affine(24.0, [6.0, 12.0, 18.0], ind)
    end

    @testset "division" begin
        @test a / 3.0 == Affine(4.0, [1.0, 2.0, 3.0], ind)
        @test a / 3   == Affine(4.0, [1.0, 2.0, 3.0], ind)
    end
end

@testset "affine arithmetic hardcode pt. 1" begin
    center  = 26.10
    dev     = [2.11, -3.03, 4.59, 1.0, -10.0]
    ind     = [1,     3,    5,    8,    10]

    @testset "power" begin
        resetLastAffineIndex()
        nCenter = 896.07645
        nDev    = [110.142, -158.166, 239.598, 52.2, -522.0, 214.86645]
        nInd    = [  1,        3,       5,      8,     10,    11]
        a = Affine(center, dev, ind)
        @test a^1 == a
        @test isapprox(a^2, Affine(nCenter, nDev, nInd); tol=1E-8)
    end

    # RINO uses CHEBYSHEV
    @testset "inverse" begin
        resetLastAffineIndex()
        nCenter = 0.06305953707868751
        nDev    = [-0.00839043, 0.0120488, -0.0182522, -0.00397651, 0.0397651, 0.0407272]
        nInd    = [ 1,          3,          5,          8,         10,        11]
        a = Affine(center)
        @test inv(a) == Affine(1 / center)
        a = Affine(center, dev, ind)
        @test isapprox(inv(a), Affine(nCenter, nDev, nInd); tol=1E-7)
        resetLastAffineIndex()
        a = Affine(center, dev, ind)
        @test isapprox(a^(-1), Affine(nCenter, nDev, nInd); tol=1E-7)
    end

    @testset "inverse equivalencies" begin
        resetLastAffineIndex()
        a     = Affine(center, dev, ind)
        inva1 = inv(a)
        resetLastAffineIndex()
        a     = Affine(center, dev, ind)
        inva2 = 1/a
        resetLastAffineIndex()
        a     = Affine(center, dev, ind)
        inva3 = a^(-1)
        @test inva1 == inva2 == inva3
    end

    @testset "sine" begin
        resetLastAffineIndex()
        nCenter = 6.0322966737821453
        nDev = [2.1099999999999999,
                -3.0299999999999998,
                4.5899999999999999,
                1,
                -10,
                20.189433933491976]
        nInd = [1, 3, 5, 8, 10, 11]
        a = Affine(center, dev, ind)
        @test isapprox(sin(a), Affine(nCenter, nDev, nInd); tol=1E-8)
    end

    @testset "cosine" begin
        resetLastAffineIndex()
        nCenter = -6.456133558502529
        nDev = [-2.1099999999999999,
                3.0299999999999998,
                -4.5899999999999999,
                -1,
                10,
                19.945823920451314]
        nInd = [1, 3, 5, 8, 10, 11]
        a = Affine(center, dev, ind)
        @test isapprox(cos(a), Affine(nCenter, nDev, nInd); tol=1E-8)
    end
end

@testset "affine arithmetic hardcode pt. 2" begin
    @testset "test mult. w/ unmatching indexes" begin
        resetLastAffineIndex()
        c1 = 1.34
        d1 = [0.21, 0.13]
        i1 = [1,    2]
        a1 = Affine(c1, d1, i1)
        c2 = 2.61
        d2 = [0.30, 0.09]
        i2 = [2,    3]
        a2 = Affine(c2, d2, i2)
        res = a1*a2
        act = Affine(3.5168999999999997,
                     [0.54809999999999992,
                      0.74130000000000007,
                      0.1206,
                      0.11309999999999999],
                     Vector(1:4))
        @test isapprox(res, act; tol=1E-8)
    end

    @testset "test mult. w/ diff. length" begin
        resetLastAffineIndex()
        c1 = 1.34
        d1 = [0.21, 0.13]
        i1 = [1,    2]
        a1 = Affine(c1, d1, i1)
        c2 = 2.61
        d2 = [0.30, -0.09, 0.17]
        i2 = [1,     2,    3]
        a2 = Affine(c2, d2, i2)
        res = a1*a2
        act = Affine(3.52305,
                     [0.95009999999999994,
                      0.21870000000000001,
                      0.22780000000000003,
                      0.15305000000000002],
                     Vector(1:4))
        @test isapprox(res, act; tol=1E-8)
    end
    
    @testset "test mult. w/ no common indexes" begin
        resetLastAffineIndex()
        c1 = 1.34
        d1 = [0.21, 0.13]
        i1 = [1,    2]
        a1 = Affine(c1, d1, i1)
        c2 = 2.61
        d2 = [0.30, 0.09]
        i2 = [3,    4]
        a2 = Affine(c2, d2, i2)
        res = a1*a2
        act = Affine(3.4974,
                     [0.5481
                      0.3393
                      0.402
                      0.1206
                      0.1326],
                     Vector(1:5))
        @test isapprox(res, act; tol=1E-8)
    end

    @testset "affine power w/ refl. n=2,3,4,5" begin
        resetLastAffineIndex()
        a = Affine(1.34, [0.21, -0.13, 0.09], [1, 2, 3])
        @test sameForm(a^2,
                       Affine(1.88805, [0.5628, -0.3484, 0.2412, 0.09245]);
                       tol=1E-8)
        @test sameForm(a^3,
                       Affine(2.7766959445512018,
                              [1.170057, -0.724321, 0.501453, 0.37270605544879931]);
                       tol=1E-8)
        @test sameForm(a^4,
                       Affine(4.226146847006234,
                              [2.2292508, -1.3800124, 0.9553932, 1.0242591629937676]);
                       tol=1E-8)
        @test sameForm(a^5,
                       Affine(6.6037146874882033,
                              [4.0897813341, 
                               -2.5317693973, 
                               1.7527634289, 
                               2.3946316179117968]);
                       tol=1E-8)

        a = -a
        @test sameForm(a^2,
                       Affine(1.88805, [0.5628, -0.3484, 0.2412, -0.09245]);
                       tol=1E-8)
        @test sameForm(a^3,
                       Affine(-2.7766959445512018,
                              [-1.170057, 0.724321, -0.501453, 0.37270605544879931]);
                       tol=1E-8)
        @test sameForm(a^4,
                       Affine(4.226146847006234,
                              [2.2292508, -1.3800124, 0.9553932, -1.0242591629937676]);
                       tol=1E-8)
        @test sameForm(a^5,
                       Affine(-6.6037146874882033,
                              [-4.0897813341, 
                               2.5317693973, 
                               -1.7527634289, 
                               2.3946316179117968]);
                       tol=1E-8)
    end
end

@testset "affine arithmetic gen." begin
    a₀ = rand(MIN .. MAX)
    b₀ = rand(MIN .. MAX)

    # xy = x₀ŷ₀ + ½∑ᴺᵢxᵢyᵢ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + [(∑ᴺᵢ|xᵢ|)(∑ᴺᵢ|yᵢ|) - ½∑ᴺᵢ|xᵢyᵢ|]μₖ
    @testset "multiplication" begin
        resetLastAffineIndex()
        acoeff = [rand(MIN .. MAX) for i in 1:4]
        bcoeff = [rand(MIN .. MAX) for i in 1:4]
        ainds  = Vector(1:4)
        binds  = Vector(1:4)
        a = Affine(a₀, acoeff, ainds)
        b = Affine(b₀, bcoeff, binds)
        res = a*b
        act = Affine(a₀*b₀ + 0.5*(acoeff' * bcoeff),
                     [(b₀*acoeff) .+ (a₀*bcoeff); 
                      sum(abs.(acoeff))*sum(abs.(bcoeff)) - 0.5*sum(abs.(acoeff .* bcoeff))],
                     Vector(1:5))
        @test res == act
    end
end

 #=
 # Computes the new noise coefficients, given two collections of noise
 #
 # Example: given function f
 #  ind1 = [ 2,   4,   6, ...]
 #  dev1 = [10.2, 5.4, 3.1,..] corresponding to 10.2ϵ₂ + 5.4ϵ₄ + 3.1ϵ₆ + ...
 #
 #  ind2 = [ 3,   4,    6, ...]
 #  dev2 = [72.0, 2.1, 27.5...] corresponding to 72.0ϵ₃ + 2.1ϵ₄ + 27.5ϵ₆ + ...
 #
 #  returns [(2, f(10.2, 0)), (3, f(0, 72.0)), (4, f(5.4,2.1)), (6, f(3.1, 27.5)),...]
 #  corresponding f(10.2, 0))ϵ₂ + f(0, 72.0)ϵ₃ + f(5.4,2.1)ϵ₄ + f(3.1, 27.5)ϵ₆ + ...
 #
 # Returns:
 #  An array, with each index ii a tuple with first and second entries corresponding 
 #  to index ii and deviation ii respectively
=#

function getGroupSolCoeffs(
    f::Function, ind1::Vector, ind2::Vector ,dev1::Vector, dev2::Vector
)
    mm   = max(last(ind1), last(ind2))
    rg   = Array(1:mm)
    addSol = Vector{Union{Nothing,AbstractFloat}}(nothing, mm)
    for ii in rg
        idx1 = findfirst(x -> x == ii, ind1)
        idx2 = findfirst(x -> x == ii, ind2)
        if(idx1 != nothing && idx2 != nothing)
            addSol[ii] = f(dev1[idx1], dev2[idx2])
        elseif(idx1 != nothing)
            addSol[ii] = f(dev1[idx1], 0)
        elseif(idx2 != nothing)
            addSol[ii] = f(0, dev2[idx2])
        end
    end
    return sort(
        filter(x -> x[2] != nothing, tuple.(rg,addSol)), 
        by = y -> y[1]
    )
end

@testset "affine arithmetic automatic" begin
    center1 = rand(Float64) + rand(0:99)
    n1 = rand(6:10)
    dev1 = rand(Float64,n1) .+ rand(-9:9, n1)
    ind1 = sort(shuffle(Array(1:20))[1:n1])
    a1 = Affine(center1,dev1,ind1)

    center2 = rand(Float64) + rand(0:99)
    n2 = rand(6:10)
    dev2 = rand(Float64,n2) .+ rand(-9:9, n2)
    ind2 = sort(shuffle(Array(1:20))[1:n2])
    a2 = Affine(center2,dev2,ind2)

    rn1 = rand(Float64) + rand(0:99)

    @testset "affine-affine addition" begin
        aAddn = a1 + a2
        @test aAddn[0] ≈ center1 + center2
        addnSol = getGroupSolCoeffs(+,ind1,ind2,dev1,dev2)
        @test length(aAddn) == length(aAddn.indexes) == length(addnSol)
        addnComp = [(aAddn.indexes[ii], dev) for (ii, dev) in enumerate(aAddn.deviations)]
        for ((compIdx, compDev), (solIdx, solDev)) in tuple.(addnComp,addnSol)
            @test compIdx == solIdx
            @test compDev ≈ solDev
        end
    end

    @testset "affine-affine substraction" begin
        aSubt = a1 - a2
        @test aSubt[0] ≈ center1 - center2
        subtSol = getGroupSolCoeffs(-,ind1,ind2,dev1,dev2)
        @test length(aSubt) == length(aSubt.indexes) == length(subtSol)
        subtComp = [(aSubt.indexes[ii], dev) for (ii, dev) in enumerate(aSubt.deviations)]
        for ((compIdx, compDev), (solIdx, solDev)) in tuple.(subtComp,subtSol)
            @test compIdx == solIdx
            @test compDev ≈ solDev
        end
    end

    @testset "affine + constant" begin
        aAddcr = a1 + rn1
        @test aAddcr[0] == center1 + rn1
        @test length(aAddcr) == length(aAddcr.indexes) == n1
        for ii in 1:n1
            @test aAddcr[ii] ≈ dev1[ii]
            @test aAddcr.indexes[ii] == ind1[ii]
        end
    end

    @testset "constant + affine" begin
        aAddcl = rn1 + a1
        @test aAddcl[0] == rn1 + center1
        @test length(aAddcl) == length(aAddcl.indexes) == n1
        for ii in 1:n1
            @test aAddcl[ii] ≈ dev1[ii]
            @test aAddcl.indexes[ii] == ind1[ii]
        end
    end

    @testset "-affine" begin
        aNeg = -a2
        @test aNeg[0] == -center2
        @test length(aNeg) == length(aNeg.indexes) == n2
        for ii in 1:n2
            @test aNeg[ii] ≈ -dev2[ii]
            @test aNeg.indexes[ii]≈ ind2[ii]
        end
    end

    @testset "affine - constant" begin
        aSubtcr = a1 - rn1
        @test aSubtcr[0] == center1 - rn1
        @test length(aSubtcr) == length(aSubtcr.indexes) == n1
        for ii in 1:n1
            @test aSubtcr[ii] ≈ dev1[ii]
            @test aSubtcr.indexes[ii] == ind1[ii]
        end
    end

    @testset "constant - affine" begin
        aSubcl = rn1 - a1
        @test aSubcl[0] == rn1 - center1
        @test length(aSubcl) == length(aSubcl.indexes) == n1
        for ii in 1:n1
            @test aSubcl[ii] ≈ dev1[ii]
            @test aSubcl.indexes[ii] == ind1[ii]
        end
    end

    @testset "affine * constant" begin
        aMultcr = a1 * rn1
        @test aMultcr[0] == center1 * rn1
        @test length(aMultcr) == length(aMultcr.indexes) == n1
        for ii in 1:n1
            @test aMultcr[ii] ≈ dev1[ii] * rn1
            @test aMultcr.indexes[ii] == ind1[ii]
        end
    end

    @testset "constant * affine" begin
        aMultcl = rn1 * a1
        @test aMultcl[0] == rn1 * center1
        @test length(aMultcl) == length(aMultcl.indexes) == n1
        for ii in 1:n1
            @test aMultcl[ii] ≈ rn1 * dev1[ii]
            @test aMultcl.indexes[ii] == ind1[ii]
        end
    end

    # xy = x₀ŷ₀ + ½∑ᴺᵢxᵢyᵢ + ∑ᴺᵢ(xᵢy₀+yᵢx₀)ϵᵢ + [(∑ᴺᵢ|xᵢ|)(∑ᴺᵢ|yᵢ|) - ½∑ᴺᵢ|xᵢyᵢ|]μₖ
    # TODO: fix the tests
    @testset "affine * affine" begin
        aMult = a1 * a2
        #@test aMult[0] ≈ center1*center2 + 0.5*(dev1'*dev2)
        multSol = getGroupSolCoeffs((x,y) -> (x*center2 + y*center1) ,ind1,ind2,dev1,dev2)
        @test length(aMult) == length(aMult.indexes) == length(multSol) + 1
        for (ii, (idx, dev)) in enumerate(multSol)
            @test aMult[ii] ≈ dev
            @test aMult.indexes[ii]  == idx
        end
        kk = length(multSol) + 1
        #@test aMult[kk] ≈ (sum(abs.(dev1)) * sum(abs.(dev2))) - 0.5*sum(abs.(dev1 .* dev2))
        @test aMult.indexes[kk] == getLastAffineIndex()
    end
end

@testset "affine vector and matrix operations" begin
    centers = [32.1, 27.3, 58.0, -10.6, 72.8, -61.0]
    devs    = [[0.1, -0.2, 1.5, -2.0],
               [10.0, 0.5, 1.0], 
               [-3.33, 9.0, -1.5, 5.25],
               [0.2, -0.33, 1.2],
               [-1.0, 0.7],
               [0.9, -4.2]]
    inds    = [[1, 3, 4, 5],
               [1, 4, 6],
               [2, 3, 5, 6],
               [3, 4, 5],
               [1, 2],
               [3, 4]]
    a1     = Affine(centers[1], devs[1], inds[1])
    a2     = Affine(centers[2], devs[2], inds[2])
    a3     = Affine(centers[3], devs[3], inds[3])
    a4     = Affine(centers[4], devs[4], inds[4])
    a5     = Affine(centers[5], devs[5], inds[5])
    a6     = Affine(centers[6], devs[6], inds[6])

    @testset "affine vector operations" begin
        @test sameForm([a1 a2] * [a3; a4], [a1*a3 + a2*a4])
        @test sameForm([a1 a2; a3 a4] * [a5; a6], [a1*a5 + a2*a6; a3*a5 + a4*a6])
    end
end

