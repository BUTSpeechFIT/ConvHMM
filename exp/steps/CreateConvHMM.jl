
using ArgParse
using ConvHMM
using BSON
using LinearAlgebra
using MarkovModels
using Random
using YAML

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--prior-strength", "-p"
            help = "strength of the prior, i.e. pseudo-counts"
            arg_type = Int
            default = 1
        "--filter-order", "-f"
            help = "order of the filter"
            arg_type = Int
            default = 5
        "--n-utts", "-n"
            help = "use `n` random utterances to compute the statistics"
            arg_type = Int
            default = 100
        "--seed", "-s"
            help = "seed of the random generator"
            arg_type = Int
            default = 1234
         "--verbose", "-v"
            help = "enable debug messages"
            action = :store_true
        "uttids"
            help = "list of training utterances"
            arg_type = String
            required = true
        "feadir"
            help = "features directory"
            arg_type = String
            required = true
        "conf-hmm"
            help = "HMM configuration file"
            arg_type = String
            required = true
        "units"
            help = "list of \"unit\" for which to create a pdf"
            arg_type = String
            required = true
        "outdir"
            help = "output directory where will be stored the HMMs and the emissions"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end


const args = getargs()
const uttids = readlines(args["uttids"])
const feadir = args["feadir"]
const confhmm = args["conf-hmm"]
const units = readlines(args["units"])
const outdir = args["outdir"]

run(`mkdir -p $outdir`)

if args["verbose"] ENV["JULIA_DEBUG"] = Main end

########################################################################
# Create the HMMs

function getstate(fsm, sid, pdfindex)
    if sid == "start"
        return initstate(fsm)
    elseif sid == "end"
        return finalstate(fsm)
    else
        if sid in keys(fsm.states)
            return fsm.states[sid]
        else
            return addstate!(fsm, pdfindex = pdfindex)
        end
    end
end

@info "creating HMMs..."

conf = open(confhmm, "r") do f YAML.load(f) end

const hmms = Dict{String, FSM}()
pdfcount = 0
for unit in units
    label, type = split(unit)
    unitconf = conf[type]

    @debug "creating HMM for unit $unit with topology type $type"

    fsm = FSM()
    for arc in unitconf["topology"]
        src = getstate(fsm, arc["start_id"], pdfcount + 1)
        if isemitting(src) && src.pdfindex > pdfcount && ! unitconf["sharedpdf"]
            global pdfcount += 1
        end

        dest = getstate(fsm, arc["end_id"], pdfcount + 1)
        if isemitting(dest) && dest.pdfindex > pdfcount && ! unitconf["sharedpdf"]
            global pdfcount += 1
        end

        link!(fsm, src, dest, log(arc["trans_prob"]))
    end
    if unitconf["sharedpdf"] global pdfcount += 1 end

    hmms[label] = fsm |> weightnormalize!
end
@debug "number of created pdf: $pdfcount"

bson(joinpath(outdir, "hmms.bson"), Dict(:hmms => hmms))

########################################################################
# Create the emissions

function getstats(rng, uttids, n, D)
    x̂ = zeros(D)
    x̂² = zeros(D)
    N = 0.
    for i in 1:n
        path = joinpath(feadir, uttids[rand(rng, 1:length(uttids))] * ".bson")
        data = BSON.load(path)[:data]
        N += size(data, 2)
        x̂ .+= dropdims(sum(data, dims = 2), dims = 2)
        x̂² .+= dropdims(sum(data.^2, dims = 2), dims = 2)
    end
    μ̂ = x̂ ./ N
    μ̂, (x̂² ./ N) - μ̂.^2
end

const S = pdfcount
const K = args["filter-order"]
const strength = args["prior-strength"]

# Check the dimension of the features
data = BSON.load(joinpath(feadir, uttids[1] * ".bson"))[:data]
const D = size(data, 1)

seed = args["seed"]
@debug "seeding the random generator with $seed"
const rng = Random.seed!(seed)

@info "estimate the statistics from the training data..."
μ̂, σ̂² = getstats(rng, uttids, args["n-utts"], D)

@info "creating the emissions"
μ₀s = [begin
    μ₀ = zeros(K+1)
    μ₀[end] = μ̂[d]
    μ₀
end for d in 1:D]
Σ₀s = [begin
    Σ₀ = Matrix{eltype(μ₀s[1])}(I, K+1, K+1) ./ strength
    Σ₀[end, end] = σ̂²[d]
    Σ₀
end for d in 1:D]
a₀ = strength
b₀ = strength
emissions = [[ARNormal1D(K, μ₀s[d], Σ₀s[d], a₀, b₀) for i in 1:S] for d in 1:D];

save(joinpath(outdir, "emissions_0.bson"), emissions)

