module ConvHMM

using ExpFamilyDistributions
using PaddedViews
using MarkovModels

export Regressors1D

include("regressor.jl")

export ARNormal1D

export elbo
export accstats_λ
export update_λ!
export accstats_h
export update_h!

include("arnormal.jl")

end

