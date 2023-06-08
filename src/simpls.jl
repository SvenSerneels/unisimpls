@doc """
Univariate SIMPLS, Jacobians and prediction intervals


Parameters:
-----------

`n_components`: int, min 1. Note that if applied on data, n_components shall
    take a value <= minimum(size(x_data))

`verbose`: Bool: to print intermediate set of columns retained

`centre`: How to internally centre data. Accepts strings ("mean","median"),
    functions (e.g. Statistics.mean) or a location vector.

`scale`: How to internally scale data. Accepts strings ("std"),
        functions (e.g. Statistics.std) or a scale vector. Enter "none" both for
        centre and scale to work with the raw data.

`copy`: Bool, whether to copy data

`all_components`: bool. `true` will return a matrix B that contains all vectors
    of regression coefficients based on [0,1,...,n_components] components

Values:
-------
The mutable struct called by SIMPLS() will be populated with model results, such
    as coef_ (regression coefficients), x_scores_, etc., as well as estimated
    locations and scales for both X and y.


Examples
--------

The module is consistent with the ScikitLearn API, e.g.

    import ScikitLearn.GridSearch:GridSearchCV
    gridsearch = GridSearchCV(unisimpls.SIMPLS(), Dict(
                    :n_components => collect(1:4), :verbose => false))
    ScikitLearn.fit!(gridsearch,X,y)
    gridsearch.best_params_
    ScikitLearn.predict(gridsearch.best_estimator_,Xt)

Example of a SIMPLS call without kernel:

    simreg = unisimpls.SIMPLS()
    unisimpls.set_params_dict!(simreg,Dict(:n_components => 3, :all_components => true))
    fit!(simreg,X,y)
    simreg.x_weights_
    predict(simreg,Xn)

Written by Sven Serneels.

References
----------

[to be completed]

""" ->
@with_kw mutable struct SIMPLS <: BaseRegressor
    n_components::Int = 1
    verbose::Bool = true
    centre = "mean"
    scale = "none"
    copy::Bool = true
    all_components = true
    return_uncertainty = true
    store_jacobians = true
    significance = .05
    X_Types_Accept = [DataFrame,Array{Number,2},Array{Float64,2},
                    Array{Float32,2},Array{Int32,2},Array{Int64,2},
                    Array{Union{Missing, Float64},2}]
    y_Types_Accept = [DataFrame,DataFrameRow{DataFrame,DataFrames.Index},
                    Array{Number,1},Array{Float64,1},Array{Float32,1},
                    Array{Int32,1},Array{Int64,1},Array{Union{Missing, Float64},1}]
    X0 = nothing
    y0 = nothing
    Xs_ = nothing
    ys_ = nothing
    x_loadings_ = nothing
    x_scores_ = nothing
    coef_ = nothing
    coef_scaled_ = nothing
    all_coeffs_ = nothing
    intercept_ = nothing
    all_intercepts_ = nothing
    x_ev_ = nothing
    y_ev_ = nothing
    fitted_ = nothing
    fit_pi_ = nothing
    all_fits_ = nothing
    all_fitted = nothing
    residuals_ = nothing
    all_residuals_ = nothing
    x_Rweights_ = nothing
    x_Rweights_unscaled_ = nothing
    x_Vloadings_ = nothing
    x_Vloadings_unscaled_ = nothing
    colret_ = nothing
    x_loc_ = nothing
    y_loc_ = nothing
    x_sca_ = nothing
    y_sca_ = nothing
    x_names = nothing
    y_name = nothing
    dbds_ = nothing
    dbdy_ = nothing
    dbdX_ = nothing
    varbs_ = nothing
    uncys_ = nothing
end

function SIMPLS(n_components;kwargs...)

    SIMPLS(n_components=n_components;kwargs...)

end

@declare_hyperparameters(SIMPLS, [:n_components,:centre, :scale])

@doc """

    Dummy function equivalent to directly creating a SNIPLS struct

    """ ->
function call(;kwargs...)

    self = SIMPLS()
    if length(kwargs) > 0
        allkeys = keys(kwargs)
        for j in allkeys
            setfield!(self,j,kwargs[j])
        end #for
    end #if
    return(self)
end #snipls

@doc """

    Fit SIMPLS model to data X and y.

    """ ->
function fit!(self::SIMPLS,X,y)

    if typeof(self.centre)==String
        if self.centre in ["mean","median"]
            self.centre = getfield(Statistics,Symbol(self.centre))
        else
            @assert self.centre=="none" "Only supported strings for median:" * "\n" *
                 "'mean', 'median', 'none'" * "\n" *
                 "Alternatively pass a function"
            # other location estimators can be included
        end
    end

    if typeof(self.scale)==String
        if self.scale in ["std"]
            self.scale = getfield(Statistics,Symbol(self.scale))
        else
            @assert self.scale=="none" "Only supported strings for scale:" * "\n" *
                 "'std','none'" * "\n" *
                 "Alternatively pass a function"
        end
    end

    X_Type = typeof(X)
    y_Type = typeof(y)
    @assert X_Type in self.X_Types_Accept "Supply X data as DataFrame or Matrix"
    @assert y_Type in self.y_Types_Accept "Supply y data as DataFrame or Vector"

    if self.copy
        setfield!(self,:X0,deepcopy(X))
        setfield!(self,:y0,deepcopy(y))
    end

    X0 = X
    y0 = y

    if y_Type == DataFrame
        ynames = true
        y_name = names(y0)
        y0 = y[:,1]
    else
        ynames = false
    end

    if X_Type == DataFrame
        Xnames = true
        X_name = names(X0)
        X0 = Matrix(X0)
    else
        Xnames = false
        X_name = nothing
    end

    allr = nothing
    allintercept = nothing
    allfit = nothing
    allB = nothing

    (n,p) = size(X0)
    ny = size(y0,1)
    @assert ny == n "Number of cases in X and y needs to agree"

    centring_X = autoscale(X,self.centre,self.scale)
    Xs= centring_X.X_as_
    mX = centring_X.col_loc_
    sX = centring_X.col_sca_

    centring_y = autoscale(y0,self.centre,self.scale)
    ys= centring_y.X_as_
    my = centring_y.col_loc_
    sy = centring_y.col_sca_

    Xs = Matrix(Xs)
    ys = ys
    # S = Xs' * Xs
    s = Xs' * ys

    nys = sum(ys.^2)

    #fit
    P,RR,R,T,V,VV,B,allB = _fit_simpls(self.n_components,n,p,Xs,ys,s,self.all_components,self.verbose)
    colret = Xnames

    if self.return_uncertainty
        if self.verbose
            print("Calculating Jacobians")
        end
        dbds = _fit_dbds(self.n_components,n,p,Xs,s,P,R,RR,T,VV,allB)
        dbdy = dbds * Xs'
        if self.store_jacobians
            setfield!(self,:dbds_, dbds)
            setfield!(self,:dbdy_, dbdy)
            setfield!(self,:dbdX_, dbds * kron(Diagonal(I,p),ys'))
        end
    end

    if self.all_components
        B_rescaled = (sy./sX)' .* allB
    else
        B_rescaled = (sy./sX)' .* B
    end

    yp_rescaled = X0*B_rescaled

    if self.all_components
        if self.centre == "none"
            intercept = zeros(1,self.n_components)
        else
            intercept = mapslices(self.centre,y .- yp_rescaled,dims=1)
        end
    elseif self.centre == "none"
        intercept = 0
    else
        intercept = mean(y .- yp_rescaled)
    end

    yfit = yp_rescaled .+ intercept
    r = y .- yfit
    if self.all_components
        allfit = yfit
        yfit = yfit[:,self.n_components]
        allr = r
        r = r[:,self.n_components]
        allB = B_rescaled
        B_rescaled = B_rescaled[:,self.n_components]
        allintercept = intercept
        intercept = intercept[:,self.n_components]
    end

    if self.return_uncertainty
        if self.scale == "none"
            @warn "Uncertainty calculations assume scaled data, results may be off"
        end
        r_scaled = ys .- Xs*B
        σ = std(r_scaled)
        σₓ = std(Xs .- T*P')
        # varb for scaled inputs
        varb = σ^2*dbdy*dbdy'
        factor = Diagonal(kron(Diagonal(I,p),ys')*(kron(Diagonal(I,p),ys')'))
        varb += σₓ*(dbds*factor*dbds')
        varb = Diagonal(varb)
        setfield!(self,:varbs_,varb)
        uncy = sqrt.(diag(1/n*Xs*varb*Xs') .+ σₓ^2 * norm(B)^2)
        uncy *= quantile.(Normal(),1-self.significance/2)
        yfitpi = hcat(yfit .- sy*uncy, yfit .+ sy*uncy)
        setfield!(self,:uncys_,uncy)
        setfield!(self,:fit_pi_,yfitpi)
    end

    setfield!(self,:x_Rweights_,R)
    setfield!(self,:x_Rweights_unscaled_,RR)
    setfield!(self,:x_loadings_,P)
    setfield!(self,:x_Vloadings_,V)
    setfield!(self,:x_Vloadings_unscaled_,VV)
    setfield!(self,:x_scores_,T)
    setfield!(self,:coef_,B_rescaled)
    setfield!(self,:all_coeffs_,allB)
    setfield!(self,:coef_scaled_,B)
    setfield!(self,:intercept_,intercept)
    setfield!(self,:all_intercepts_,allintercept)
    setfield!(self,:fitted_,yfit)
    setfield!(self,:all_fits_,allfit)
    setfield!(self,:residuals_,r)
    setfield!(self,:all_residuals_,allr)
    setfield!(self,:ys_,ys)
    setfield!(self,:x_loc_,mX)
    setfield!(self,:y_loc_,my)
    setfield!(self,:x_sca_,sX)
    setfield!(self,:y_sca_,sy)
    if Xnames
        setfield!(self,:x_names,X_name)
    end
    if ynames
        setfield!(self,:y_name,y_name)
    end

    return(self)

end


@doc """

    Fit SPLS model to data X and y and only return the regression
    coefficients.

    """ ->
function fit(self::SIMPLS,X,y)

    if self.X0 == nothing
        fit!(self,X,y)
    end

    return(self.coef_)

end

@doc """

    Predict responses for new predictor data.

    """ ->
function predict(self::SIMPLS,Xn)

    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    return(Xn * self.coef_ .+ self.intercept_[:,1])

end

@doc """

    Prediction interval for predicted responses
    Inputs
        SIMPLS object
        Xn, new data of consistent dimensions (1 or more rows)
        siginificance, optional, significance level at which to
            construct prediction interval, if different from the one specified
            in SIMPLS object
    """ ->
function predint(self::SIMPLS,Xn,significance=nothing)

    if significance == nothing
        significance = self.significance
    end
    Xn, nxn, pxn = _predict_check(self,Xn)
    n,p = size(Xn)
    n₀ = length(self.fitted_)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    yp = Xn * self.coef_ .+ self.intercept_[:,1]
    uncy = diag(sqrt.(1/n₀*Xn*self.varb_*Xn'))
    uncy *= quantile.(Normal(),1-significance/2)
    return hcat(yp .- uncy, yp .+ uncy)

end

@doc """

    For a model with `n_components` = k, predict the dependent variable for all
    settings 1:k.

    """ ->
function predict_all(self::SIMPLS,Xn)

    @assert self.all_components "To predict full set of components, flag `all_components` needs to be `true`"
    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    return((Xn * self.all_coeffs_ .+ self.all_intercepts_))

end

@doc """

    Transform new predictor data to estimated scores.

    """ ->
function transform(self::SIMPLS,Xn)

    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    Xnc = autoscale(Xn,self.x_loc_,self.x_sca_).X_as_
    return(Xnc*self.x_Rweights_)

end

@doc """

    Get all settings from an existing SPLS struct (also the ones not
        declared as parameters to ScikitLearn)

    """ ->
function get_all_params(self::T, param_names=[]::Array{Any}) where{T}

    if length(param_names)==0

        params_dict = type2dict(self)

    else

        params_dict = Dict{Symbol, Any}()

        for name::Symbol in param_names
            params_dict[name] = getfield(self, name)
        end

    end

    params_dict

end

@doc """

    ScikitLearn similar function to set parameters in an existing SPLS
        struct, yet takes a Dict as an argument.

    Compare:
    ScikitLearn.set_params!(lnrj,eta=.3)
    spls.set_params_dict!(lnrj,Dict(:eta => .3))

    """ ->
function set_params_dict!(self::T, params; param_names=nothing) where {T}

    for (k, v) in params

        if param_names !== nothing && !(k in param_names)

            throw(ArgumentError("An estimator of type $T was passed the invalid hyper-parameter $k. Valid hyper-parameters: $param_names"))

        end

        setfield!(self, k, v)

    end

    if self.verbose
        print(self)
    end

end

clone_param(v::Any) = v # fall-back

function is_classifier(self::unisimpls.SIMPLS) return(false) end

@doc """

    ScikitLearn compatible function to clone an existing SPLS struct.

    """ ->
function clone(self::T) where {T}

    kw_params = Dict{Symbol, Any}()

    # cloning the values is scikit-learn's default behaviour. It's ok?

    for (k, v) in get_params(self) kw_params[k] = clone_param(v) end

    return T(; kw_params...)

end

function get_params(self::SIMPLS)
    return spls.get_all_params(self)
end

function set_params!(self::SIMPLS; params...)
    return spls.set_params_dict!(self,params)
end

function Base.copy!(newcopy::SIMPLS, self::SIMPLS)
    # shallow copy - used below
    for f in fieldnames(typeof(newcopy))
        setfield!(newcopy, f, getfield(self, f))
    end
end
