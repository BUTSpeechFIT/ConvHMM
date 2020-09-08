module ConvHMM

using BSON
using ExpFamilyDistributions
using PaddedViews
using MarkovModels

export Regressors1D

include("regressor.jl")

export ARNormal1D

export accstats_λ
export accstats_h
export elbo
export predict
export save
export update_λ!
export update_h!

include("arnormal.jl")

end

