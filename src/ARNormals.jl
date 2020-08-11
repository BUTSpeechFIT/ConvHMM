# AutoRegressive Normal Model.
#

using LinearAlgebra
using PaddedViews


export ARNormal1D
export ARNormal
export meanfilter
export Regressors1D


#######################################################################
# Regressors1D is an iterable to access dynamically the context of a
# given sample.
#

struct Regressors1D
    K::Int64
    x::PaddedView
end

function Regressors1D(K::Int64, x::Array{T}) where T
    px = PaddedView(T(0.), x, (1-K:length(x),))
    Regressors1D(K, px)
end

Base.length(r::Regressors1D) = length(parent(r.x))

function Base.getindex(r::Regressors1D, t::Int)
    retval = r.x[t-r.K:t]
    retval[end] = 1 # "bias" regressor
    retval
end

function Base.iterate(r::Regressors1D, state = nothing)
    K = r.K
    if isnothing(state)
        T = eltype(r.x)
        state = (Vector{T}(undef, K+1), 1)
    end

    buffer, t = state
    if t > length(r) return nothing end

    buffer[:] = r.x[t-K:t]
    buffer[end] = 1 # bias regressor
    return (buffer, (buffer, t+1))
end

#######################################################################
# 1D AutoRegressive Normal.

struct ARNormal1D{K} <: Model
    # IIR Filter
    h::ConjugateParameter{<:Normal}

    # Noise precision
    λ::ConjugateParameter{<:Gamma}

    function ARNormal1D(h, λ) where K
        μ₀, _ = stdparam(h.prior)
        model = new{length(μ₀) - 1}(h, λ)
        h.stats = (X) -> stats1D_h(model, X)
        λ.stats = (X) -> stats1D_λ(model, X)
        return model
    end
end

function ARNormal1D(
    μ₀::Vector{T},
    Σ₀::Matrix{T},
    μ::Vector{T},
    Σ::Matrix{T},
    a₀::Real,
    b₀::Real,
    a::Real,
    b::Real
) where T <: AbstractFloat
    ARNormal1D(
        ConjugateParameter(Normal(μ₀, Σ₀), Normal(μ, Σ)),
        ConjugateParameter(Gamma([T(a₀)], [T(b₀)]), Gamma([T(a)], [T(b)])),
    )
end

function ARNormal1D(
    T::Type;
    order::Integer,
    initσ::Real = 0.0,
    pseudocounts::Real = 1.0
)
    # Parameters of the prior over the filter `h`
    μ₀ = zeros(T, order+1)
    Σ₀ = T(1/pseudocounts) * Matrix{T}(I, order+1, order+1)

    # Initial parameters of the posterior over the filter `h`
    μ = μ₀ .+ T(initσ) .* randn(T)
    Σ = deepcopy(Σ₀)

    # We use `pseudocounts` as the default parameterization of the
    # Gamma prior/posterior
    ARNormal1D(μ₀, Σ₀, μ, Σ, pseudocounts, pseudocounts, pseudocounts,
               pseudocounts)
end

meanfilter(model::ARNormal1D{K}) where K = reverse(mean(model.h.posterior)[1:K])


function (model::ARNormal1D{K})(x::Vector{F}) where {F <: AbstractFloat, K}
    px = PaddedView(0., x, (1-K:length(x), ))
    regressors = Regressors1D{F, K}(px)

    # Expectation of the natural parameters of the model.
    E_λ, E_lnλ = gradlognorm(model.λ.posterior)
    E_Th = gradlognorm(model.h.posterior)

    N = length(px) - K
    llh = zeros(T, N)
    for t in 1:T
        r = vec(regressors[n])
        stats = vcat(r * px[n], -.5 * vec(r * r'))
        llh[n] = -.5 * (log(2π) - E_lnλ + E_λ * px[n]^2) + E_λ * dot(E_Th, stats)
    end
    llh
end


function stats1D_h(model::ARNormal1D{K}, x::Vector{T}) where {T <: AbstractFloat, K}
    px = PaddedView(0., x, (1-K:length(x), ))
    regressors = Regressors1D{T, K}(px)
    E_λ, E_lnλ = gradlognorm(model.λ.posterior)
    retval = zeros(T, (K + 1) * (K + 2), length(x))
    N = length(px) - K
    for n in 1:N
        r = vec(regressors[n])
        retval[:, n] = vcat(E_λ * r * px[n], -.5 * E_λ * vec(r * r'))
    end
    retval
end


function stats1D_λ(model::ARNormal1D{K}, x::Vector{T}) where {T <: AbstractFloat, K}
    px = PaddedView(0., x, (1-K:length(x), ))
    regressors = Regressors1D{T, K}(px)
    E_Th = gradlognorm(model.h.posterior)
    retval = zeros(T, 2, length(x))
    N = length(px) - K
    for n in 1:N
        r = vec(regressors[n])
        stats = vcat(r * px[n], -.5 * vec(r * r'))
        retval[:, n] = [-.5 * px[n]^2 + dot(E_Th, stats), .5]
    end
    retval
end

#######################################################################
# ARNormal: concatenation of D independent ARNormal1D

struct ARNormal{D, K} <: Model
    filters::Vector{ARNormal1D}

    function ARNormal(filters::Vector{ARNormal1D{K}}) where K
        model = new{length(filters), K}(filters)

        # We override the stats
        for (d, filter) in enumerate(filters)
            filter.h.stats = data -> stats_h(model, d, data)
            filter.λ.stats = data -> stats_λ(model, d, data)
        end
        model
    end
end

function ARNormal(μ₀::Vector{T}, Σ₀::Matrix{T}, μ::Vector{T},
                  Σ::Matrix{T}, a₀::Real, b₀::Real, a::Real, b::Real,
                  dim::Integer) where T <: AbstractFloat
    ARNormal([ARNormal1D(μ₀, Σ₀, deepcopy(μ), deepcopy(Σ), a₀, b₀,
                         deepcopy(a), deepcopy(b)) for i in 1:dim])
end

function ARNormal(T = Float64; dim::Integer, order::Integer, initσ::Real = 0.0,
                    pseudocounts::Real = 1.0)
    # Parameters of the prior over the filter `h`
    μ₀ = zeros(T, order+1)
    Σ₀ = T(1/pseudocounts) * Matrix{T}(I, order+1, order+1)

    # Initial parameters of the posterior over the filter `h`
    μ = μ₀ .+ T(initσ) .* randn(T)
    Σ = deepcopy(Σ₀)

    # We use `pseudocounts` as the default parameterization of the
    # Gamma prior/posterior
    ARNormal(μ₀, Σ₀, μ, Σ, pseudocounts, pseudocounts, pseudocounts,
             pseudocounts, dim)
end

getconjugateparams(model::ARNormal) = vcat([m.h for m in model.filters],
                                           [m.λ for m in model.filters])

function (model::ARNormal)(X::Matrix{T}) where T <: AbstractFloat
    retval = zeros(T, size(X, 2))
    for (d, m) in enumerate(model.filters)
        retval .+= m(X[d, :])
    end
    retval
end

function stats_h(model::ARNormal, d::Integer, X::Matrix{T}) where T <: AbstractFloat
    stats1D_h(model.filters[d], X[d, :])
end

function stats_λ(model::ARNormal, d::Integer, X::Matrix{T}) where T <: AbstractFloat
    stats1D_λ(model.filters[d], X[d, :])
end

