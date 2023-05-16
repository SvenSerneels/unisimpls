@doc """

    Autoscale a data matrix

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::Matrix,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators to be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                              Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            # Further scale estimates to be included
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    (n,p) = size(X)

    if pre_estimated_loc
        x_loc_ = locest
        if length(size(x_loc_))==1
            x_loc_ = x_loc_'
        end
    else
        if locest == "none"
            x_loc_ = zeros(1,p)
        else
            x_loc_ = mapslices(x -> locest(x),X,dims=1)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
        if length(size(x_sca_))==1
            x_sca_ = x_sca_'
        end
    else
        if scalest == "none"
            x_sca_ = ones(1,p)
        else
            x_sca_ = mapslices(x -> scalest(x),X,dims=1)
        end
    end

    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (X - ones(n,1)*x_loc_) ./ (ones(n,1)*x_sca_)
    end
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

@doc """

    Autoscale a data frame

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::DataFrame,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators can be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    (n,p) = size(X)
    x_names = names(X)

    if pre_estimated_loc
        x_loc_ = locest
        if length(size(x_loc_))>1
            x_loc_ = x_loc_'
        end
    else
        if locest == "none"
            x_loc_ = zeros(1,p)
        else
            x_loc_ = mapslices(x -> locest(x),Array(X),dims=1)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
        if length(size(x_sca_))>1
            x_sca_ = x_sca_'
        end
    else
        if scalest == "none"
            x_sca_ = ones(1,p)
        else
            x_sca_ = mapslices(x -> scalest(x),Array(X),dims=1)
        end
    end
    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (Array(X) .- x_loc_) ./ (x_sca_)
        X_as_ = DataFrame(X_as_,:auto)
    end
    rename!(X_as_,[Symbol(n) for n in x_names])
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

@doc """

    Autoscale a data vector

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::Vector,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators to be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},Float64,Int64,Float32,Int32]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},Float64,Int64,Float32,Int32]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    if pre_estimated_loc
        x_loc_ = locest
    else
        if locest == "none"
            x_loc_ = 0
        else
            x_loc_ = locest(X)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
    else
        if scalest == "none"
            x_sca_ = 1
        else
            x_sca_ = scalest(X)
        end
    end
    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (X .- x_loc_) ./ x_sca_
    end
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

function _norms(X)

    """
    Casewise norms of a matrix
    """
        return(mapslices(norm,X,dims=2))
end

function ss(dd, p, args...)

    """
    Computes the spatial sign radial function
    args:
      dd: vector of distances
      p: dimension of original data
      args flag to be able to pass on n - has no effect
    returns:
      xi: radial function
    """

    prec = 1e-10
    dd = max.(dd,prec)
    xi = 1 ./ dd
    return(xi,nothing,nothing)

end

function hampel(x,cutoffs)

    """
    Computes the Hampel redescending function

    args:
        x: input as Vector
        cutoffs, onsisting of: probct, hampelb, hampelr, cutoff values for (reweighting,
        harsher reweighting, rejection). If x~Dist than good values for these
        constants can be based on the quantiles of Dist.

    values:
        wx: reweighted x
    """

    probct,hampelb,hampelr = cutoffs
    wx = deepcopy(x)
    wx[findall(x .<= probct)] .= 1
    wx[findall((x .> probct) .& (x .<= hampelb))] .= probct./abs.(x[
            findall((x .> probct) .& (x .<= hampelb))])
    wx[findall((x .> hampelb) .& (x .<= hampelr))] .=
            probct * (hampelr .- (x[findall((x .> hampelb)
            .& (x .<= hampelr))])) ./
            ((hampelr - hampelb) * abs.(x[findall((x .> hampelb) .&
            (x .<= hampelr))]))
    wx[findall(x .> hampelr)] .= 0

    return(wx)

end

function linear_redescending(dd, p,n,d_hmed=nothing,cutoff=nothing)

    """
    # Computes the Linear redescending radial function
    args:
      dd: vector of distances
      p: number of variables in original data
      n: number of rows in original data
    returns:
      xi: radial function
    note:
      Cutoffs are based on hmed/hmad as in the paper, designed to avoid implosion
      beakdown. When n <= (n+p+1)/2, med/mad are used instead of hmed/hmad, with
      the notion that implosion breakdown is possible in such a case.
    """

    if ((d_hmed == nothing) | (cutoff==nothing))
        h = Int(floor((n + p + 1) / 2))
        if n <= h
            h = Int(floor((n/ 2)))
        end
        dWH = dd.^(2/3)
        dWH_hmed = sort(dWH,dims=1)[h]
        dWH_hmad = sort(abs.(dWH .- dWH_hmed),dims=1)[h]
        d_hmed = dWH_hmed^(3/2)
        cutoff = (dWH_hmed + 1.4826 * dWH_hmad)^(3/2)
    end
    idxmid = findall((dd .> d_hmed) .& (dd .<= cutoff))
    idxhigh = findall(dd .> cutoff)
    xi = ones(n)# xi = ones((n,1))
    xi[idxmid] = 1 .- (dd[idxmid,:] .- d_hmed) ./ (cutoff .- d_hmed)
    xi[idxhigh] .= 0
    return(xi,d_hmed,cutoff)

end

function mad(x,c=1.4826)

    """
    Consitency corrected median absolute deviation estimator.
    """

    return(c*median(abs.(x .- median(x))))

end

function minmaxscale(xm,a=0,b=1;kwargs...)

    """
    Min/Max scaling onto the interval [a,b]
    """

    if length(size(xm))==1
        xm = reshape(xm,(size(xm)[1],1))
    end
    n,p = size(xm)

    if length(kwargs)==0
        minx = mapslices(minimum,xm,dims=1)
        maxx = mapslices(maximum,xm,dims=1)
    elseif length(kwargs)<2
        throw("If ranges unspecified both min and max need to be provided")
    else
        minx = kwargs[:minx]
        maxx = kwargs[:maxx]
    end

    delta = maxx - minx
    xm = a .+ ((xm.-minx)./delta).*(b-a)

    return(xm,minx,maxx)

end

function radbas(n,b=1;kwargs...)

    """
    Radial basis function transformation
    """

    if b!=1
        nr,nc = size(n)
        n .*= (b*ones(1,nc))
    end
    a=exp.(-(n.*n))

end

function dck(Xd;kwargs...)

    """
    Double centred kernel transformation
    """

    if !(:ṁ in keys(kwargs))
        ṁ = mapslices(mean,Xd,dims=1)
    else
        ṁ = kwargs[:ṁ]
    end

    if !(:m̈ in keys(kwargs))
        m̈ = mean(Xd)
    else
        m̈ = kwargs[:m̈]
    end

    n,p = size(Xd)
    if n>p
        return(Xd .- ṁ[1:p] .- ṁ' .+ m̈)
    elseif p>n
        return(Xd .- ṁ .- ṁ[1] .+ m̈)
    else
        return(Xd .- ṁ .- ṁ' .+ m̈)
    end

end
