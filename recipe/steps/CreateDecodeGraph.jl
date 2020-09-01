# Build a decoding graph from a set of models, a lexicon and a language
# model (LM).
#

using ArgParse
using BSON
using MarkovModels

# Disambiguation symbol to distinguish between word state and
# phone/character state
const DIS_SYM = "wrd"

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--add-sil", "-s"
            help = "add silence add the beginning/end of decoding graph (used only if lm is not specified)"
            action = :store_true
        "--lm", "-l"
            help = "Language Model graph to use (if not provided, build a flat LM)"
            default = ""
        "lexicon"
            help = "mapping word -> pronunciation"
            arg_type = String
            required = true
        "hmms"
            help = "unit hmms"
            arg_type = String
            required = true
        "decgraph"
            help = "output decoding graph"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end

const args = getargs()
const lmfile = args["lm"]
const lexiconfile = args["lexicon"]
const hmmsfile = args["hmms"]
const out = args["decgraph"]

run(`mkdir -p $(dirname(out))`)

@info "loading the hmms"
const hmms = BSON.load(hmmsfile)[:hmms]

@info "loading the pronunciations"
const pronuns = Dict{String, Vector{FSM}}()
open(lexiconfile, "r") do f
    for line in eachline(f)
        tokens = split(line)
        word = "$DIS_SYM:$(tokens[1])"
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

function build_flat_lm(lexicon, addsil::Bool)
    lm = FSM()
    init = initstate(lm)
    final = finalstate(lm)
    if addsil
        init = addstate!(lm, label = "sil")
        link!(lm, initstate(lm), init)
        final = addstate!(lm, label = "sil")
        link!(lm, final, finalstate(lm))
    end
    for w in keys(lexicon)
        s = addstate!(lm, label = w)
        link!(lm, init, s)
        link!(lm, s, final)
    end
    link!(lm, final, init)
    lm |> weightnormalize!
end

const lm = begin
    if lmfile == ""
        @info "no LM provided, building a flat LM"
        build_flat_lm(lexicon, args["add-sil"])
    else
        @info "using a flat LM"
        lm = BSON.load(lmfile)[:lm]
    end
end

@info "compiling the decoding graph"
decodegraph = compose!(compose!(lm, lexicon), hmms) |> removenilstates!
bson(out, Dict(:decodegraph => decodegraph))

