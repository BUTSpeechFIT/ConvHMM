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

    # we return a shallow copy of the buffer to avoid issue if the user
    # does not copy the data himself.
    return (copy(buffer), (buffer, t+1))
end

#######################################################################
# Iterator over overlapping frames of a signal.

struct FrameIterator{T}
    signal::Vector{T}
    framelength::Int64
    hopsize::Int64
    window::Vector{T}
end

function Base.length(it::FrameIterator)
    if length(it.signal) <= it.framelength
        return 0
    end
    1 + (length(it.signal) - it.framelength) ÷ it.hopsize
end

function Base.iterate(it::FrameIterator, state::Int64=1)
    if state > length(it)
        return nothing
    end
    framestart = (state - 1) * it.hopsize + 1
    frameend = framestart + it.framelength - 1
    (it.signal[framestart:frameend], state + 1)
end

# Return an iterator over the frames of the signal `x`
function frames(x::Vector{T}, sr::Real, t::Real, Δt::Real, window::Function) where T <:AbstractFloat
    N = Int64(sr * t)
    FrameIterator{T}(x, N, Int64(sr * Δt), window(N))
end

#######################################################################
# Hann window

function hannwindow(N::Int64)
    Ωₙ = 2π / N
    x = range(-N/2, N/2, length = N)
    cos.((Ωₙ / 2) .* x).^2
end

