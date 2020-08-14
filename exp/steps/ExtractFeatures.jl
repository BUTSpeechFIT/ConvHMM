
using ArgParse
using Distributed

# BUT specific package to work with the SGE cluster
# See https://github.com/BUTSpeechFIT/BUTSGEManager
using BUTSGEManager

function getargs()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--sampling-rate", "-s"
            help = "sampling rate of the input file in Hz"
            arg_type = Int
            default = 16000
        "--frame-duration", "-d"
            help = "Duration in seconds of the analysis window"
            arg_type = Real
            default = 0.025
        "--frame-rate", "-r"
            help = "Hop size of the analysis window in seconds"
            arg_type = Real
            default = 0.01
        "--n-filters", "-n"
            help = "number of filters in the filter-bank"
            arg_type = Int
            default = 26
        "--low-freq", "-l"
            help = "low cut-off frequency of the filter-bank in Hz"
            arg_type = Real
            default = 80
        "--high-freq", "-c"
            help = "high cut-off frequency of the filter-bank in Hz"
            arg_type = Real
            default = 8000
        "--jobs", "-j"
            help = "number of jobs to run in parallel"
            arg_type = Int
            default = 4
        "--jobs-args", "-a"
            help = "arguments to pass to the parallel engine"
            arg_type = String
            default = ""
        "outdir"
            help = "output directory where will be stored the features"
            arg_type = String
            required = true
        "scps"
            help = "input scp files"
            arg_type = String
            nargs='+'
            required = true
    end
    args = parse_args(s)
end


const args = getargs()
const outdir = args["outdir"]
const scps = args["scps"]


# Make sure the directory exists
run(`mkdir -p $outdir`)

@info "Initializing the workers"
addprocs(SGEManager(args["jobs"]), args = args["jobs-args"],
         exeflags = "--project=$(Base.active_project())")
@everywhere using SpeechFeatures
@everywhere using WAV
@everywhere using BSON
@everywhere const args = $args
@everywhere const outdir = $outdir

@info "building the features extractor"
@everywhere const extractor = LogMelSpectrum(
    fftlen = fftlen_auto,
    srate = args["sampling-rate"],
    frameduration = args["frame-duration"],
    framestep = args["frame-rate"],
    removedc = true,
    preemphasis = 0.97,
    dithering = 0.,
    windowfn = hann,
    windowpower = 1.0,
    nfilters = args["n-filters"],
    lofreq = args["low-freq"],
    hifreq = args["high-freq"]
)

totalcount = 0
for scp in scps
    @info "extracting the features from $scp"

    count = @distributed (a, b) -> a + b for line in readlines(scp)
        uttid, fname = split(line)

        channels, srate = wavread(fname, format="double")
        channels *= typemax(Int16)
        s = channels[:, 1]

        data = s |> extractor

        bson(joinpath(outdir, uttid * ".bson"),
             Dict(:data => data, :framerate => args["frame-rate"]))

        # to count how many recordings were processed
        1
    end
    global totalcount += 1
end

@info "successfuly processed $totalcount files"


