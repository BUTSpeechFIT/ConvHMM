# Discriminant AutoRegressive Normal
#

struct DARNormal1D{K}
    arnormals::ARNormal1DSet{K}

    # Prior/posterior over `ξ`, the IIR filter modeling the noise
    ξprior::Normal
    ξposterior::Normal
end

#######################################################################
# Constructors

function DARNormal1D(
    arnormals::ARNormal1DSet{K},
    u₀::Vector{T},
    V₀::Matrix{T};
    u::Union{Vector{T}, Missing} = missing,
    V::Union{Matrix{T}, Missing} = missing
) where {K, T <: AbstractFloat}
    u = ismissing(u) ? copy(u₀) : u
    V = ismissing(V) ? copy(V₀) : V
    DARNormal1D{K}(arnormals, Normal(u₀, V₀), Normal(u, V))
end

#######################################################################
# Expected log-likelihood

function loglikelihood(
    model::DARNormal1D{K},
    x::Vector{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    η_hs = vcat([gradlognorm(m.hposterior)' for m in model.arnormals]...)
    η_λs = vcat([gradlognorm(m.λposterior)' for m in model.arnormals]...)
    η_ξ = gradlognorm(model.ξposterior)

    ms = [mean(m.hposterior) for m in model.arnormals]
    u = mean(model.ξposterior)

    # Sufficient statistics for h
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_hξ = vcat(s1, -.5 * s2)

    iter = zip(eachrow(η_hs), eachrow(η_λs), ms)
    vcat([(hcat(-.5 * x.^2 .+ stats_hξ' * (η_h + η_ξ)  .- s2' *vec(m * u'),
                .5 * ones(T, length(x))) * η_λ)'
          for (η_h, η_λ, m) in iter]...) .- .5 * log(2π)
end

function loglikelihood(
    models::Vector{DARNormal1D{K}},
    X::Matrix{T}
) where {K, T<:AbstractFloat}
    reduce(+, [loglikelihood(models[d], X[d, :]) for d in 1:size(X, 1)])
end

#######################################################################
# Posterior predictive
#
# NOTE: we use the MAP of precision parameters and we integrate over the
#       the filters (h_1, h_2, ...)

function predict(
    model::DARNormal1D{K},
    x::Vector{T}
) where {K, T<:AbstractFloat}

    @warn "erroneous computation"

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
    models::Vector{DARNormal1D{K}},
    x::Vector{T}
) where {K, T<:AbstractFloat}
    vcat([predict(models[s], x)' for s in 1:length(models)]...)
end

"""
    predict(model, X)

Return the logarithm of the posterior predictive distribution for each
frame of `X`
"""
function predict(
    models::Vector{Vector{DARNormal1D{K}}},
    X::Matrix{T}
) where {K, T<:AbstractFloat}
    reduce(+, [predict(models[d], X[d, :]) for d in 1:size(X, 1)])
end

########################################################################
# Accumulate statistics

function accstats_h(
    model::ARNormal1D{K},
    x::Vector{T},
    γ::Vector{T},
    ξ::Vector{T}
) where {K, T<:AbstractFloat}

    r = Regressors1D(K, x)

    # Sufficient statistics
    λ = mean(model.λposterior)
    s1 = hcat([(xₜ- ξ' * x̂ₜ) * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_h = λ .* vcat(s1, -.5 * s2)

    stats_h * γ
end

function accstats_h(
    model::DARNormal1D{K},
    x::Vector{T},
    γ::Matrix{T}
) where {K, T<:AbstractFloat}
    ξ = mean(model.ξposterior)
    [accstats_h(m, x, γ[d, :], ξ) for (d, m) in enumerate(model.arnormals)]
end

function accstats_ξ(
    model::ARNormal1D{K},
    x::Vector{T},
    γ::Vector{T}
) where {K, T<:AbstractFloat}

    r = Regressors1D(K, x)

    # Sufficient statistics
    h = mean(model.hposterior)
    λ = mean(model.λposterior)[1]
    s1 = hcat([λ * (xₜ- h' * x̂ₜ) * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([λ * vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_ξ =  vcat(s1, -.5 * s2)

    stats_ξ * γ
end

function accstats_ξ(
    model::DARNormal1D{K},
    x::Vector{T},
    γ::Matrix{T}
) where {K, T<:AbstractFloat}
    reduce(+, [accstats_ξ(m, x, γ[s, :]) for (s, m) in enumerate(model.arnormals)])
end

function accstats_λ(
    model::DARNormal1D{K},
    x::Vector{T},
    γ::Matrix{T}
) where {K, T<:AbstractFloat}
    r = Regressors1D(K, x)

    η_hs = vcat([gradlognorm(m.hposterior)' for m in model.arnormals]...)
    η_ξ = gradlognorm(model.ξposterior)

    ms = [mean(m.hposterior) for m in model.arnormals]
    u = mean(model.ξposterior)

    # Sufficient statistics for h
    s1 = hcat([xₜ * x̂ₜ for (xₜ, x̂ₜ) in zip(x, r)]...)
    s2 = hcat([vec(x̂ₜ* x̂ₜ') for x̂ₜ in r]...)
    stats_hξ = vcat(s1, -.5 * s2)

    [hcat(-.5 * x.^2 .+ stats_hξ' * (η_h + η_ξ)  .- s2' *vec(m * u'),
          .5 * ones(T, length(x)))' * γₛ
     for (η_h, m, γₛ) in zip(eachrow(η_hs), ms, eachrow(γ))]
end

function accstats_λ(
    models::Vector{DARNormal1D{K}},
    x::Vector{T},
    γ::Matrix{T}
) where {K, T<:AbstractFloat}
    [accstats_λ(m, x, γ[d, :]) for (d, m) in enumerate(models)]
end

#######################################################################
# Utilities

function save(
    fname::AbstractString,
    model::Vector{DARNormal1D{K}}
) where K
    bson(fname, Dict(:model => model, :order=>K, :type=>typeof(model)))
end

