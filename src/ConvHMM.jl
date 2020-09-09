module ConvHMM

using BSON
using ExpFamilyDistributions
using LinearAlgebra
using PaddedViews
using MarkovModels

export Regressors1D

include("regressor.jl")

export ARNormal1D
export ARNormal1DSet
export DARNormal1D

export accstats_λ
export accstats_ξ
export accstats_h
export elbo
export loglikelihood
export predict
export save
export update_λ!
export update_h!

include("arnormal.jl")
include("darnormal.jl")

end

