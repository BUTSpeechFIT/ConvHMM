
using ArgParse
using BSON
using ConvHMM
using Distributed
using ExpFamilyDistributions
using Glob
using MarkovModels
using NaturalSort

# BUT specific package to work with the SGE cluster
# See https://github.com/BUTSpeechFIT/BUTSGEManager
using BUTSGEManager

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--epochs", "-e"
            help = "number of epochs to train"
            arg_type = Int
            default = 1
        "--jobs", "-j"
            help = "number of jobs to run in parallel"
            arg_type = Int
            default = 4
        "--jobs-args", "-a"
            help = "arguments to pass to the parallel engine"
            arg_type = String
            default = ""
        "uttids"
            help = "list of utterance ids to train on"
            arg_type = String
            required = true
        "feadir"
            help = "features directory"
            arg_type = String
            required = true
        "alidir"
            help = "alignment graphs directory"
            arg_type = String
            required = true
        "modeldir"
            help = "directory where was created the initial model"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end

const args = getargs()
const uttids = readlines(args["uttids"])
const feadir = args["feadir"]
const alidir = args["alidir"]
const outdir = args["modeldir"]
const epochs = args["epochs"]

# Make sure the directory exists
run(`mkdir -p $outdir`)

########################################################################
# Setup the workers
@info "initializing the workers"
addprocs(SGEManager(args["jobs"]), args = args["jobs-args"],
         exeflags = "--project=$(Base.active_project())")

@everywhere using BSON
@everywhere using ExpFamilyDistributions
@everywhere using MarkovModels
@everywhere using ConvHMM
@everywhere const args = $args
@everywhere const feadir = $feadir
@everywhere const alidir = $alidir

########################################################################
# Functions to accumulate statistics and to update the parameters

# Sum dictionaries fo statistics
@everywhere function mergestats(
    s1::Dict{Int64, Dict{Int64, Vector{Float64}}},
    s2::Dict{Int64, Dict{Int64, Vector{Float64}}}
)
    retval = Dict{Int64, Dict{Int64, Vector{Float64}}}()
    for d in keys(s1)
        retval[d] = Dict{Int64, Vector{Float64}}()
        for s in keys(s1[d])
            retval[d][s] = s1[d][s] + s2[d][s]
        end
   end
   retval
end

@everywhere function accumulate_h(
    model::Vector{Vector{ARNormal1D{K}}},
    X::Matrix{T},
    γ::Matrix{T}
) where {K, T <: AbstractFloat}

    D = length(model)
    S = length(model[1])

    accstats = Dict{Int64, Dict{Int64, Vector{Float64}}}()
    for d in 1:D
        accstats[d] = Dict{Int64, Vector{Float64}}()
        for (s, stats) in zip(1:S, accstats_h(model[d], X[d, :], γ))
            accstats[d][s] = stats
        end
    end
    accstats
end

@everywhere function accumulate_λ(
    model::Vector{Vector{ARNormal1D{K}}},
    X::Matrix{T},
    γ::Matrix{T}
) where {K, T <: AbstractFloat}

    D = length(model)
    S = length(model[1])

    accstats = Dict{Int64, Dict{Int64, Vector{Float64}}}()
    for d in 1:D
        accstats[d] = Dict{Int64, Vector{Float64}}()
        for (s, stats) in zip(1:S, accstats_λ(model[d], X[d, :], γ))
            accstats[d][s] = stats
        end
    end
    accstats
end

function update_h!(
    model::Vector{Vector{ARNormal1D{K}}},
    stats::Dict{Int64, Dict{Int64, Vector{T}}},
) where {K, T <: AbstractFloat}

    D = length(model)
    S = length(model[1])
    for d in 1:D
        for s in 1:S
            m = model[d][s]
            s = stats[d][s]
            η₀ = naturalparam(m.hprior)
            update!(m.hposterior, η₀ .+ s)
        end
    end
end

function update_λ!(
    model::Vector{Vector{ARNormal1D{K}}},
    stats::Dict{Int64, Dict{Int64, Vector{T}}},
) where {K, T <: AbstractFloat}

    D = length(model)
    S = length(model[1])
    for d in 1:D
        for s in 1:S
            m = model[d][s]
            s = stats[d][s]
            η₀ = naturalparam(m.λprior)
            update!(m.λposterior, η₀ .+ s)
        end
    end
end

########################################################################
# Training
#
# NOTE: the training is an EM-like algorithm. For the M-step, we
# alternately update the filters (h) or the precision parameters (λ)

# KL-divergence post/prior of the parameters of the model
function kldiv_post(model)
    KL = 0.
    for d in 1:length(model)
        for m in model[d]
            KL += kldiv(m.hposterior, m.hprior)
            KL += kldiv(m.λposterior, m.λposterior)
        end
    end
    KL
end

# Retrieve the last check-point
fnames = [basename(fname) for fname in glob(joinpath(outdir, "emissions_*bson"))]
fnames = sort(fnames, lt=natural)
epochno = replace(replace(fnames[end], ("emissions_" => "")), (".bson" => ""))
start = parse(Int64, epochno) + 1

if start > 1
    @info "found $(fnames[end]), starting training from epoch $start"
end

@everywhere reducer(a, b) = (mergestats(a[1], b[1]), a[2] + b[2], a[3] + b[3])
for e in start:epochs
    path = joinpath(outdir, "emissions_$(e-1).bson")
    @everywhere emissions = ConvHMM.load($path)

    @everywhere epoch = $e
    stats, totll, totN = @distributed reducer for uttid in uttids
        println(uttid)
        X = BSON.load(joinpath(feadir, uttid * ".bson"))[:data]
        ali = BSON.load(joinpath(alidir, uttid * ".bson"))[:ali]

        S = length(emissions[1])
        N = size(X, 2)

        # E-step
        lnαβ, totll = αβrecursion(ali, emissions(X))
        γ_sparse = resps(ali, lnαβ, dense = false)
        γ = zeros(S, N)
        for idx in keys(γ_sparse) γ[idx, :] = γ_sparse[idx] end

        if e % 2 == 1
            accstats = accumulate_h(emissions, X, γ)
        else
            accstats = accumulate_λ(emissions, X, γ)
        end

        (accstats, totll, N)
    end

    𝓛 = (totll - kldiv_post(emissions)) / totN
    @info "epoch $(e)/$(epochs) 𝓛 = $(round(𝓛, digits = 3))"

    if e % 2 == 1
        update_h!(emissions, stats)
    else
        update_λ!(emissions, stats)
    end

    save(joinpath(outdir, "emissions_$(e).bson"), emissions)
end

