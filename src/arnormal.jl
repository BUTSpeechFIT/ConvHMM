# Autoregressive-Normal
#

#######################################################################
# Model structure and constructor

"""
    struct ARNormal
        hprior::Normal
        hposterior::Normal
        λprior::Gamma
        λposterior::Gamma
    end

AutoRegressive Normal distribution of order `K`. When `K` is `0`, the
distribution is a univariate Normal distribution. Note that `hprior`
and `hposterior` are `K+1` multivariate Normal distribution.

Constructor:

    ARNormal1D(μ₀, Σ₀, a₀, b₀[, μ = missing, Σ = missing, a = missing, b = missing])

The filter order `K` is set to `length(μ₀) - 1`.
"""
struct ARNormal1D{K}
    # Prior/posterior over `h`, the IIR filter
    hprior::Normal
    hposterior::Normal

    # Prior/posterior over `λ`, noise precision
    λprior::Gamma
    λposterior::Gamma
end

function ARNormal1D(
    μ₀::Vector{T},
    Σ₀::Matrix{T},
    a₀::Real,
    b₀::Real;
    μ::Union{Vector{T}, Missing} = missing,
    Σ::Union{Matrix{T}, Missing} = missing,
    a::Union{Real, Missing} = missing,
    b::Union{Real, Missing} = missing
) where T <: AbstractFloat
    K = length(μ₀) - 1
    μ = ismissing(μ) ? copy(μ₀) : μ
    Σ = ismissing(Σ) ? copy(Σ₀) : Σ
    a = ismissing(a) ? a₀ : a
    b = ismissing(b) ? b₀ : b

    a₀ = T(a₀)
    b₀ = T(b₀)
    a = T(a)
    b = T(b)
    ARNormal1D{K}(Normal(μ₀, Σ₀), Normal(μ, Σ), Gamma([a₀], [b₀]), Gamma([a], [b]))
end

"""
    const ARNormal1DSet{K} = Vector{ARNormal1D{K}} where K

Set of AutoRegressive Normal distributions.
"""
const ARNormal1DSet{K} = Vector{ARNormal1D{K}} where K

#######################################################################
# Expected log-likelihood

"""
    loglikelihood(model::ARNormal1D{K}, x::Vector{T})
    loglikelihood(models::ARNormal1DSet{K}, x::Vector{T})
    loglikelihood(models::Vector{ARNormal1DSet{K}}, X::Matrix{T})

Evaluate the expectation of the log-likelihood w.r.t. the variational
posterior. When the model is a (vector of) `ARNomralSet`, the
log-likelihood per component is returned.
"""
function loglikelihood(
    model::ARNormal1D{K},
    fiter::FrameIterator{T}
) where {K, T<:AbstractFloat}
    # Expectation of the natural parameters
    η_h = gradlognorm(model.hposterior)
    η_λ = gradlognorm(model.λposterior)

    llh = T[]
    for x in fiter
        r = Regressors1D(K, x)
        # Sufficient statistics for h
        s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
        s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
        stats_h = vcat(s1, -.5 * s2)

        # Sufficient statistics for λ
        stats_λ = hcat(-.5 * x.^2 .+ stats_h' * η_h, .5 * ones(T, length(x)))

        push!(llh, sum(stats_λ * η_λ .- .5 * log(2π)))
    end
    llh
end

function loglikelihood(
    model::ARNormal1D{K},
    x::Vector{T}
) where {K, T <: AbstractFloat}
    r = Regressors1D(K, x)

    # Expectation of the natural parameters
    η_h = gradlognorm(model.hposterior)
    η_λ = gradlognorm(model.λposterior)

    # Sufficient statistics for h
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = vcat(s1, -.5 * s2)

    # Sufficient statistics for λ
    stats_λ = hcat(-.5 * x.^2 .+ stats_h' * η_h, .5 * ones(T, length(x)))

    stats_λ * η_λ .- .5 * log(2π)
end

function loglikelihood(
    models::ARNormal1DSet{K},
    fiter::FrameIterator{T}
) where {K, T<:AbstractFloat}
    η_hs = vcat([gradlognorm(model.hposterior)' for model in models]...)
    η_λs = vcat([gradlognorm(model.λposterior)' for model in models]...)

    llh = Vector{Vector{T}}()
    for x in fiter
        r = Regressors1D(K, x)

        # Sufficient statistics for h
        s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
        s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
        stats_h = vcat(s1, -.5 * s2)

        iter = zip(eachrow(η_hs * stats_h), eachrow(η_λs))
        llh_t = vcat([(hcat(-.5 * x.^2 .+ row, .5 * ones(T, length(x))) * η_λ)'
                      for (row, η_λ) in iter]...) .- .5 * log(2π)
        llh_t = dropdims(sum(llh_t, dims = 2), dims = 2)
        push!(llh, llh_t)
    end
    hcat(llh...)
end

function loglikelihood(
    models::ARNormal1DSet{K},
    x::Vector{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    η_hs = vcat([gradlognorm(model.hposterior)' for model in models]...)
    η_λs = vcat([gradlognorm(model.λposterior)' for model in models]...)

    # Sufficient statistics for h
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = vcat(s1, -.5 * s2)

    iter = zip(eachrow(η_hs * stats_h), eachrow(η_λs))
    vcat([(hcat(-.5 * x.^2 .+ row, .5 * ones(T, length(x))) * η_λ)'
          for (row, η_λ) in iter]...) .- .5 * log(2π)
end

function loglikelihood(
    models::Vector{ARNormal1DSet{K}},
    fiters::Vector{FrameIterator{T}}
) where {K, T<:AbstractFloat}
    reduce(+, [loglikelihood(models[d], fiters[d]) for d in 1:length(fiters)])
end

function loglikelihood(
    models::Vector{ARNormal1DSet{K}},
    X::Matrix{T}
) where {K, T<:AbstractFloat}
    reduce(+, [loglikelihood(models[d], X[d, :]) for d in 1:size(X, 1)])
end

#######################################################################
# Posterior predictive
#
# NOTE: we use the MAP of precision parameters and we integrate over the
#       the filters (h_1, h_2, ...)

"""
    predict(model::ARNormal1D{K}, x::Vector{T})
    predict(model::ARNormal1DSet{K}, x::Vector{T})
    predict(model::Vector{ARNormal1DSet{K}}, x::Vector{T})

Return the logarithm of the posterior predictive distribution for each
frame `x[t]`.
"""
function predict(
    model::ARNormal1D{K},
    x::Vector{T}
) where {K, T<:AbstractFloat}

    r = Regressors1D(K, x)
    λ_map_inv = 1 / mean(model.λposterior)[1]
    m = model.hposterior.μ
    Σ = model.hposterior.Σ

    # Sufficient statistics for h
    s1 = hcat([x̂ₜ for x̂ₜ in r]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)

    ψ = s1' * m
    ϑ_inv = s2' * vec(Σ)

    prec = 1 ./ (λ_map_inv .+ ϑ_inv)

    -.5 .* prec .* (x .- ψ).^2 .+ .5 .* log.(prec) .- .5 * log(2π)
end

function predict(
    models::ARNormal1DSet{K},
    x::Vector{T}
) where {K, T<:AbstractFloat}
    vcat([predict(models[s], x)' for s in 1:length(models)]...)
end

function predict(
    models::Vector{Vector{ARNormal1D{K}}},
    X::Matrix{T}
) where {K, T<:AbstractFloat}
    reduce(+, [predict(models[d], X[d, :]) for d in 1:size(X, 1)])
end

########################################################################
# Accumulate statistics

function accstats_h(
    models::Vector{ARNormal1D{K}},
    x::Vector{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    # Sufficient statistics
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = vcat(s1, -.5 * s2)

    accstats = (resps * stats_h')

    [mean(m.λposterior) .* s for (m, s) in zip(models, eachrow(accstats))]
end

function accstats_h(
    models::ARNormal1DSet{K},
    fiter::FrameIterator{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}

    stats_h = Vector{Vector{T}}()
    for x in fiter
        r = Regressors1D(K, x)

        # Sufficient statistics
        s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
        s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
        stats_h_t = vcat(s1, -.5 * s2)

        stats_h_t = dropdims(sum(stats_h_t, dims = 2), dims = 2)
        push!(stats_h, stats_h_t)
    end
    stats_h = hcat(stats_h...)
    accstats = (resps * stats_h')

    [mean(m.λposterior) .* s for (m, s) in zip(models, eachrow(accstats))]
end

function accstats_λ(
    models::Vector{ARNormal1D{K}},
    x::Vector{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    # Sufficient statistics
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = vcat(s1, -.5 * s2)

    η_hs = vcat([gradlognorm(model.hposterior)' for model in models]...)

    accstats = Vector{Vector{T}}()
    for (hs, γ)  in zip(eachrow(η_hs * stats_h), eachrow(resps))
        λs = hcat(-.5 * x.^2 .+ hs, .5 * ones(T, length(x)))'
        s = λs * γ
        push!(accstats, s)
    end
    accstats
end

function accstats_λ(
    models::ARNormal1DSet{K},
    fiter::FrameIterator{T},
    γ::Matrix{T}
) where {K, T<:AbstractFloat}
    ssize = length(gradlognorm(models[1].λposterior)
                  )
    stats_λ = zeros(T, length(models), ssize)
    η_hs = vcat([gradlognorm(model.hposterior)' for model in models]...)
    for (t, x) in enumerate(fiter)
        r = Regressors1D(K, x)

        # Sufficient statistics
        s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
        s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
        stats_h_t = vcat(s1, -.5 * s2)

        Hs = η_hs * stats_h_t
        stats_λ_t = Vector{Vector{T}}()
        for (i, hs) in enumerate(eachrow(Hs))
            s = hcat(-.5 * x.^2 .+ hs, .5 * ones(T, length(x)))
            s = dropdims(sum(s, dims = 1), dims = 1)
            stats_λ[i, :] .+= s * γ[i, t]
        end
    end
    stats_λ
end

#######################################################################
# Utilities

function save(
    fname::AbstractString,
    model::Vector{Vector{ARNormal1D{K}}}
) where K
    bson(fname, Dict(:model => model, :order=>K, :type=>typeof(model)))
end

function load(
    fname::AbstractString,
)
    data = BSON.load(fname)
    m = data[:model]
    K = data[:order]
    T = data[:type]
    convert(T, m)
end

