module unisimpls

    using Statistics, DataFrames, Parameters, Random, Distances, Distributions, LinearAlgebra
    import ScikitLearnBase: BaseRegressor, BaseEstimator, predict, fit!, fit_transform!, @declare_hyperparameters, is_classifier, clone, transform # get_params, set_params!, transform,
    # import LinearAlgebra: diagm, norm, eigen
    import ScikitLearn
    # include("_sreg_utils.jl")
    include("_preproc_utils.jl")
    include("simpls.jl")
    include("unisimpls_algo.jl")

    export SIMPLS, autoscale, set_params_dict!, predict_all

end
