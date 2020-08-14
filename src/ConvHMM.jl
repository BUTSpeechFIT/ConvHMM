module ConvHMM

using BSON
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
export save
#export load

include("arnormal.jl")

end

