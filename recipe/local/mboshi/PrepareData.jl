using ArgParse
using Glob
using Base.Filesystem

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "rootdir"
            help = "root directory of the github repository"
            required = true
        "outdir"
            help = "root directory of the github repository"
            required = true
    end
    args = parse_args(s)
end

getuttid(path) = splitext(basename(path))[1]
getspeaker(uttid) = split(uttid, "_")[1]

function main()
    args = getargs()

    inputdir = abspath(args["rootdir"])
    outputdir = abspath(args["outdir"])

    for x in ["train", "dev"]
        @info "preparing $x set..."

        run(`mkdir -p $(joinpath(outputdir, x))`)

        @info "creating wav.scp file"
        uttids = Set()
        open(joinpath(outputdir, x, "wav.scp"), "w") do scp
            for path in glob("*wav", joinpath(inputdir, "full_corpus_newsplit", x))
                uttid = getuttid(path)
                println(scp, "$uttid $path")
                push!(uttids, uttid)
            end
        end

        @info "creating uttids file"
        open(joinpath(outputdir, x, "uttids"), "w") do f
            for u in uttids
                println(f, "$u")
            end
        end

        @info "creating speaker mapping"
        open(joinpath(outputdir, x, "utt2spk"), "w") do utt2spk
            for uttid in uttids
                spk = getspeaker(uttid)
                println(utt2spk, "$uttid $spk")
            end
        end

        @info "preparing phonetic transcription"
        open(joinpath(outputdir, x, "trans.phn"), "w") do trans
            for path in glob("*mb", joinpath(inputdir, "full_corpus_newsplit", x))
                uttid = getuttid(path)
                text = filter(l -> !isspace(l), lowercase(readline(path)))
                phns = [l for l in text]
                println(trans, "$uttid $(join(phns, " "))")
            end
        end

        @info "preparing word transcription"
        open(joinpath(outputdir, x, "trans.wrd"), "w") do trans
            for path in glob("*mb", joinpath(inputdir, "full_corpus_newsplit", x))
                uttid = getuttid(path)
                text = lowercase(readline(path))
                words = [w for w in split(text, " ")]
                println(trans, "$uttid $(join(words, " "))")
            end
        end

        @info "prepared $(length(uttids)) utterances for $x set"
    end

    @info "preparing the lang directory..."
    run(`mkdir -p $(joinpath(outputdir, "lang"))`)

    @info "creating the word level lexicon"
    lexicon = Dict()
    phones = Set()
    for path in glob("*mb", joinpath(inputdir, "full_corpus_newsplit", "all"))
        text = lowercase(readline(path))
        for w in split(text)
            for l in w push!(phones, l) end
            lexicon[w] = [l for l in w]
        end
    end
    open(joinpath(outputdir, "lang", "lexicon.wrd"), "w") do f
        for w in sort([k for k in keys(lexicon)])
            println(f, "$w $(join(lexicon[w], " "))")
        end
    end

    @info "creating the phone level lexicon (identity map)"
    open(joinpath(outputdir, "lang", "lexicon.char"), "w") do f
        for w in sort([k for k in phones])
            println(f, "$w $w")
        end
    end

    @info "creating the unit inventory"
    open(joinpath(outputdir, "lang", "units"), "w") do f
        println(f, "sil non-speech-unit")
        for p in phones
            println(f, "$p speech-unit")
        end
    end

end

main()

