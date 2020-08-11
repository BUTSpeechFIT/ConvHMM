# Regressors1D is an iterable to access dynamically the context,
# i.e. x[t-1], ..., x[t-K], of a time step.

struct Regressors1D{K}
    x::PaddedView
end

function Regressors1D(K::Int64, x::Array{T}) where T <: AbstractFloat
    px = PaddedView(T(0.), x, (1-K:length(x),))
    Regressors1D{K}(px)
end

Base.length(r::Regressors1D) = length(parent(r.x))

function Base.getindex(r::Regressors1D{K}, t::Int) where K
    retval = r.x[t-K:t]
    retval[end] = 1 # "bias" regressor
    retval
end

function Base.iterate(r::Regressors1D{K}, state = nothing) where K
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

