# Compute the Token Error Rate between to transcript files.
#

using ArgParse

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--analysis", "-a"
            help = "output analysis file"
            default = ""
        "ref"
            help = "reference transcription"
            arg_type = String
            required = true
        "hyp"
            help = "hypothesis transcription"
            arg_type = String
            required = true
    end
    args = parse_args(s)
end

function loadtrans(fname::AbstractString)
    trans = Dict{String, String}()
    open(fname, "r") do f
        for line in eachline(f)
            tokens = split(line)
            trans[tokens[1]] = join(tokens[2:end], " ")
        end
    end
    trans
end

function editdistance(reftokens, hyptokens)
    distmat = zeros(Int64, length(reftokens) + 1, length(hyptokens) + 1)

    for i in 1:length(reftokens)+1 distmat[i, 1] = i-1 end
    for j in 1:length(hyptokens)+1 distmat[1, j] = j-1 end

    for i in 2:length(reftokens)+1
        for j in 2:length(hyptokens)+1

            if reftokens[i-1] == hyptokens[j-1]
                distmat[i, j] = distmat[i-1, j-1]
            else
                sub = distmat[i-1, j-1] + 1
                insert = distmat[i, j-1] + 1
                delete = distmat[i-1, j] + 1
                distmat[i, j] = min(sub, insert, delete)
            end
        end
    end
    distmat
end

function analyze(reftokens, hyptokens, editmat)
    x, y = length(reftokens) + 1, length(hyptokens) + 1

    subs = Dict{Tuple{AbstractString, AbstractString}, Int64}()
    inserts = Dict{AbstractString, Int64}()
    deletions = Dict{AbstractString, Int64}()

    while x > 1 || y > 1
        if x >= 2 && y >= 2 && editmat[x, y] == editmat[x-1, y-1] && reftokens[x-1] == hyptokens[y-1]

            x -= 1
            y -= 1
        elseif y >= 2 && editmat[x, y] == editmat[x, y-1] + 1

            key = hyptokens[y-1]
            v = get(inserts, key, 0)
            inserts[key] = v + 1

            y -= 1
        elseif x >= 2 && y >= 2 && editmat[x, y] == editmat[x-1, y-1] + 1

            key = (reftokens[x-1], hyptokens[y-1])
            v = get(subs, key, 0)
            subs[key] = v + 1

            x -= 1
            y -= 1
        else
            key = reftokens[x-1]
            v = get(deletions, key, 0)
            deletions[key] = v + 1

            x -= 1
        end
    end
    subs, inserts, deletions
end


const args = getargs()
const analysisfile = args["analysis"]
const reffile = args["ref"]
const hypfile = args["hyp"]

@info "loading the transcriptions"
const refs = loadtrans(reffile)
const hyps = loadtrans(hypfile)


const subs = Dict{Tuple{String, String}, Int64}()
const inserts = Dict{String, Int64}()
const deletions = Dict{String, Int64}()
count_utt = 0
count_token = 0
for uttid in keys(hyps)
    ref = split(refs[uttid])
    hyp = split(hyps[uttid])

    mat = editdistance(ref, hyp)
    s, i, d = analyze(ref, hyp, mat)

    for k in keys(s)
        v = get(subs, k, 0)
        subs[k] = v + 1
    end
    for k in keys(d)
        v = get(deletions, k, 0)
        deletions[k] = v + 1
    end
    for k in keys(i)
        v = get(inserts, k, 0)
        inserts[k] = v + 1
    end

    global count_utt += 1
    global count_token += length(ref)
end

# Number of substitutions
S = reduce(+, values(subs), init = 0)
D = reduce(+, values(deletions), init = 0)
I = reduce(+, values(inserts), init = 0)

# Token Error Rate
TER = round(100 * (S + D + I) / count_token, digits = 3)

@info "TER = $TER % (substitutions: $S, deletions: $D, insertions: $I) evaluated from $count_utt utterances"

