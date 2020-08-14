
using ArgParse
using JLD2
using MarkovModels
using Distributed

# BUT specific package to work with the SGE cluster
# See https://github.com/BUTSpeechFIT/BUTSGEManager
using BUTSGEManager

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--add-sil", "-s"
            help = "add silence add the beginning/end of each utterance"
            action = :store_true
        "--jobs", "-j"
            help = "number of jobs to run in parallel"
            arg_type = Int
            default = 4
        "--jobs-args", "-a"
            help = "arguments to pass to the parallel engine"
            arg_type = String
            default = ""
        "trans"
            help = "list of word level transcription"
            arg_type = String
            required = true
        "lexicon"
            help = "mapping word -> pronunciation"
            arg_type = String
            required = true
        "hmms"
            help = "unit hmms"
            arg_type = String
            required = true
        "outdir"
            help = "output directory where will be stored the alignment graphs"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end


const args = getargs()
const trans = args["trans"]
const lexiconfile = args["lexicon"]
const hmmsfile = args["hmms"]
const outdir = args["outdir"]

@info "loading the pronunciations"
const pronuns = Dict{String, Vector{FSM}}()
open(lexiconfile, "r") do f
    for line in eachline(f)
        tokens = split(line)
        word = tokens[1]
        pronun = tokens[2:end]
        ps = get(pronuns, word, FSM[])
        push!(ps, LinearFSM(pronun))
        pronuns[word] = ps
    end
end

@info "compiling pronunciation graphs"
const lexicon = Dict{String, FSM}()
for (k, v) in pronuns
    lexicon[k] = union(v...) |> minimize!
end

# Make sure the directory exists
run(`mkdir -p $outdir`)

@info "initializing the workers"
addprocs(SGEManager(args["jobs"]), args = args["jobs-args"],
         exeflags = "--project=$(Base.active_project())", clean_output = false)
@everywhere using JLD2
@everywhere using MarkovModels
@everywhere const outdir = $outdir
@everywhere const lexicon = $lexicon
@everywhere const hmmsfile = $hmmsfile
@everywhere @load hmmsfile hmms

@info "compiling alignment graphs"
@sync @distributed for line in readlines(trans)
    tokens = split(line)
    uttid = tokens[1]
    words = tokens[2:end]
    if args["add-sil"]
        insert!(words, 1, "sil")
        push!(words, "sil")
    end

    ali = compose!(compose!(LinearFSM(words), lexicon), hmms) |> removenilstates!
    jldopen(joinpath(outdir, uttid * ".jld2"), "w") do f
        f["ali"] = ali
    end
end

