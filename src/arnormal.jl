# Autoregressive-Normal
#

struct ARNormal1D{K}
    # Prior/posterior over `h`, the IIR filter
    hprior::Normal
    hposterior::Normal

    # Prior/posterior over `λ`, noise precision
    λprior::Gamma
    λposterior::Gamma
end

#######################################################################
# Constructors

function ARNormal1D(
    K::Int64,
    μ₀::Vector{T},
    Σ₀::Matrix{T},
    a₀::Real,
    b₀::Real,
    μ::Vector{T},
    Σ::Matrix{T},
    a::Real,
    b::Real
) where T <: AbstractFloat
    ARNormal1D{K}(
        Normal(μ₀, Σ₀),
        Normal(μ, Σ),
        Gamma([T(a₀)], [T(b₀)]),
        Gamma([T(a)], [T(b)]),
    )
end
ARNormal1D(K, μ₀, Σ₀, a₀, b₀) = ARNormal1D(K, μ₀, Σ₀, a₀, b₀, deepcopy(μ₀),
                                           deepcopy(Σ₀), a₀, b₀)
ARNormal1D(K, μ₀, Σ₀, a₀, b₀, μ, Σ) = ARNormal1D(K, μ₀, Σ₀, a₀, b₀, μ, Σ, a₀, b₀)

#######################################################################
# Expected log-likelihood

function (model::ARNormal1D{K})(x::Vector{T}) where {K, T <: AbstractFloat}
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

    stats_λ * η_λ .- log(2π)
end

function (models::Vector{ARNormal1D{K}})(x::Vector{T}) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    η_hs = vcat([gradlognorm(model.hposterior)' for model in models]...)
    η_λs = vcat([gradlognorm(model.λposterior)' for model in models]...)

    # Sufficient statistics for h
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = vcat(s1, -.5 * s2)

    vcat([(hcat(-.5 * x.^2 .+ row, .5 * ones(T, length(x))) * η_λ)'
           for (row, η_λ) in zip(eachrow(η_hs * stats_h), eachrow(η_λs))]...)
end

function (models::Vector{Vector{ARNormal1D{K}}})(X::Matrix{T}) where {K, T<:AbstractFloat}
    reduce((a, b) -> a .+ b, [models[d](X[d, :]) for d in 1:size(X, 1)])
end

#######################################################################
# Accumulate statistics

function update_h!(
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

    for (m, s) in zip(models, eachrow(accstats))
        E_λ = mean(m.λposterior)
        η₀ = naturalparam(m.hprior)
        update!(m.hposterior, η₀ .+ s .* E_λ)
    end
end

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

function update_h!(
    models::Vector{Vector{ARNormal1D{K}}},
    X::Matrix{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}
    for d in 1:length(models)
        update_h!(models[d], X[d, :], resps)
    end
end

function update_λ!(
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

    for (hs, m, γ)  in zip(eachrow(η_hs * stats_h), models, eachrow(resps))
        λs = hcat(-.5 * x.^2 .+ hs, .5 * ones(T, length(x)))'
        accstats = λs * γ
        η₀ = naturalparam(m.λprior)
        update!(m.λposterior, η₀ .+ accstats)
    end
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

function update_λ!(
    models::Vector{Vector{ARNormal1D{K}}},
    X::Matrix{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}
    for d in 1:length(models)
        update_λ!(models[d], X[d, :], resps)
    end
end

#######################################################################
# Utilities

function save(
    fname::AbstractString,
    model::Vector{Vector{ARNormal1D{K}}}
) where K
    bson(fname, Dict(:model => model, :order=>K))
end

function load(
    fname::AbstractString,
)
    data = BSON.load(fname)
    m = data[:model]
    K = data[:order]
    convert(Vector{Vector{ARNormal1D{K}}}, m)
end

export trajectory

function trajectory(
    models::Vector{ARNormal1D{K}},
    x::Vector{T},
    resps::Matrix{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    E_λ = vcat([mean(model.λposterior)' for model in models]...)
    hs = vcat([mean(model.hposterior)' for model in models]...)

    # Sufficient statistics for h
    s1 = hcat([hs * x̂ₜ for x̂ₜ in r]...)

    μ = dropdims(sum(s1 .* resps, dims = 1), dims = 1)
    λ = dropdims(sum(E_λ .* resps, dims = 1), dims = 1)
    μ, λ
end


