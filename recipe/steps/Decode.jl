
using ArgParse
using BSON
using ConvHMM
using Distributed
using ExpFamilyDistributions
using Glob
using MarkovModels

# BUT specific package to work with the SGE cluster
# See https://github.com/BUTSpeechFIT/BUTSGEManager
using BUTSGEManager

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--disambig-sym", "-d"
            help = "Disambiguation symbol to distinguish word state from phone state"
            default = "wrd"
        "--jobs", "-j"
            help = "number of jobs to run in parallel"
            arg_type = Int
            default = 4
        "--jobs-args", "-a"
            help = "arguments to pass to the parallel engine"
            arg_type = String
            default = ""
        "--marginalize", "-m"
            help = "marginalize the parameters w.r.t. the variational posterior"
            action = :store_true
        "--pruning", "-p"
            help = "pruning threshold to use (a negative value means no pruning"
            arg_type = Float64
            default = -1.0
        "emissions"
            help = "emissions to use to decode"
            arg_type = String
            required = true
        "decodegraph"
            help = "decoding graph"
            arg_type = String
            required = true
        "uttids"
            help = "list of utterance ids to decode"
            arg_type = String
            required = true
        "feadir"
            help = "features directory"
            arg_type = String
            required = true
        "decodedir"
            help = "directory where was created the initial model"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end

const args = getargs()
const dis_sym = args["disambig-sym"]
const emissionsfile = args["emissions"]
const dgraphfile = args["decodegraph"]
const uttids = readlines(args["uttids"])
const feadir = args["feadir"]
const decodedir = args["decodedir"]
const marginalize = args["marginalize"]
const pruning = args["pruning"] < 0 ? nopruning : args["pruning"]

# Make sure the directory exists
run(`mkdir -p $decodedir`)

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
@everywhere const marginalize = $marginalize
@everywhere const dis_sym = $dis_sym
@everywhere const pruning = $pruning

@everywhere const emissions = ConvHMM.load($emissionsfile)
@everywhere const decgraph = BSON.load($dgraphfile)[:decodegraph]

########################################################################
# Decoding
#

# Decode one utterance
@everywhere function decode(
    emissions::Vector{Vector{T1}},
    decgraph::FSM,
    fea::Matrix{T2},
    marginalize::Bool,
    dis_sym::AbstractString
) where {T1 <: ARNormal1D, T2 <: AbstractFloat}

    p = marginalize ? predict(emissions, fea) : emissions(fea)

    # `path` is a LinearFSM
    path = viterbi(decgraph, p, pruning = pruning)

    # Browse the best path FSM to get the string of labels
    queue = State[initstate(path)]
    syms = String[]
    while ! isempty(queue)
        next = iterate(children(path, pop!(queue)))
        if isnothing(next) continue end
        label = next[1].dest.label
        if ! isnothing(label) && startswith(label, "$dis_sym:")
            sym = label[length(dis_sym)+2:end]
            push!(syms, sym)
        end
        push!(queue, next[1].dest)
    end
    syms
end

# To avoid conflict with previous run, we remove all the partial
# decoding file.
@info "cleaning the previous decoding results"
for fname in filter!(f -> startswith(f, "trans"), readdir(decodedir))
    rm(joinpath(decodedir, fname), force = true)
end

@info "decoding..."
counter = @distributed (+) for uttid in uttids
    X = BSON.load(joinpath(feadir, uttid * ".bson"))[:data]
    syms = decode(emissions, decgraph, X, marginalize, dis_sym)

    open(joinpath(decodedir, "trans.$(myid())"), "a") do f
        println(f, "$uttid $(join(syms, " "))")
    end

    1 # to count the number of utterances decoded
end


# Gather all the results in one file
@info "concatenating results"
outfile = joinpath(decodedir, "trans")
for fname in filter!(f -> startswith(f, "trans"), readdir(decodedir))
    open(outfile, "a") do f
        write(f, read(joinpath(decodedir, fname)))
    end
end

@info "sucessfully decoded $counter utterances"

