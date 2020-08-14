
using ArgParse
using ConvHMM
using Distributed
using JLD2
using MarkovModels

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
        "emissions"
            help = "emissions"
            arg_type = String
            required = true
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
        "outdir"
            help = "output directory where will be stored the model"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end

const args = getargs()
const uttids = readlines(args["uttids"])
const feadir = args["feadir"]
const alidir = args["alidir"]
const outdir = args["outdir"]
const epochs = args["epochs"]

# Make sure the directory exists
run(`mkdir -p $outdir`)

@info "initializing the workers"
addprocs(SGEManager(args["jobs"]), args = args["jobs-args"],
         exeflags = "--project=$(Base.active_project())", clean_output=false)

@everywhere using JLD2
@everywhere using MarkovModels
@everywhere using ConvHMM
@everywhere const feadir = $feadir
@everywhere const alidir = $alidir

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

@everywhere const args = $args
@load args["emissions"] emissions
@everywhere emissions = $emissions

@everywhere const D = length(emissions)
@everywhere const S = length(emissions[1])

for e in 1:epochs
    stats_h = @distributed mergestats for uttid in uttids
        println(uttid)
        X = jldopen(joinpath(feadir, uttid * ".jld2"), "r") do f
            f["data"]
        end
        ali = jldopen(joinpath(alidir, uttid * ".jld2"), "r") do f
            f["ali"]
        end
        N = size(X, 2)

        # E-step
        lnαβ, totllh = αβrecursion(ali, emissions(X))
        γ_sparse = resps(ali, lnαβ, dense = false)

        γ = zeros(S, N)
        for idx in keys(γ_sparse) γ[idx, :] = γ_sparse[idx] end

        accstats = Dict{Int64, Dict{Int64, Vector{Float64}}}()
        for d in 1:D
            accstats[d] = Dict{Int64, Vector{Float64}}()
            for (s, stats) in zip(1:S, accstats_h(emissions[d], X[d, :], γ))
                accstats[d][s] = stats
            end
        end
        accstats
    end

    # Update filters
    for d in 1:D
        for s in 1:S
            m = emissions[d][s]
            stats = accstats_h[d][s]
            η₀ = naturalparam(m.hprior)
            update!(m.hposterior, η₀ .+ stats)
        end
    end
end

exit()
@info "initializing the workers"
addprocs(SGEManager(args["jobs"]), args = args["jobs-args"],
         exeflags = "--project=$(Base.active_project())")
@everywhere using JLD2
@everywhere using MarkovModels
@everywhere using ConvHMM
@everywhere using ExpFamilyDistributions
@everywhere const feadir = $feadir
@everywhere const alidir = $alidir


for e in 1:epochs
    for uttid in uttids
        @load joinpath(feadir, uttid * ".jld2") data
        @load joinpath(alidir, uttid * ".jld2") ali

        # E-step
        lnαβ, totllh = αβrecursion(ali, models(X))
        γ = resps(ali, lnαβ, dense = true)

        break
    end
end

