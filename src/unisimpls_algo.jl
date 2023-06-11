@doc """
`unisimpls_algo` Univariate SIMPLS Algorithm and Jacobians

Can be called directly or though the scikit-learn API.

Written by Sven Serneels.

""" ->

function _fit_simpls(n_components,n,p,Xs,ys=[],s=[],all_components=false,verbose=false)

    """
    the univariate SIMPLS algorithm
    adapted from MATLAB code by Sijmen de Jong
    """
    if verbose
        print("Fitting a " * string(n_components) * " component SIMPLS model" * "\n")
    end
    n,p = size(Xs);
    T = zeros(n,n_components);
    R = zeros(p,n_components);
    RR = zeros(p,n_components);
    P = zeros(p,n_components);
    V = zeros(p,n_components);
    VV = zeros(p,n_components);
    v = zeros(p,1);
    b = zeros(p,1);
    if all_components
        B = zeros(p,n_components)
    else
        B = nothing
    end

    if length(s) == 0
        s = Xs'*ys
    end
    rh = deepcopy(s);

    for a = 1:n_components
       r = deepcopy(s)
       t = Xs * r
       normt =  sqrt(t'*t)
       t = t / normt
       rr = r / normt
       p = Xs' * t
       v = p - V[:,1:max(1,a-1)] * (V[:,1:max(1,a-1)]' * p)
       VV[:,a] = v
       v = v / sqrt(v'*v)
       s = s - v * (v' * s)
       T[:,a] = t
       P[:,a] = p
       R[:,a] = r
       V[:,a] = v
       RR[:,a] = rr
       b = RR[:,1:a]*RR[:,1:a]'*Xs'*ys
       if all_components
           B[:,a] = b
       end
    end

    return P,R,RR,T,V,VV,b,B

end

function _fit_dbds(n_components,n,p,Xs,s,P,R,RR,T,V,B)

    """
    _fit_dbds
    the univariate SIMPLS Jacobian matrices with respect to
    the vector of cross covariance with static predictor
    """

    n,p = size(Xs);

    if n_components>min(n,p)
        throw("Max # LV cannot exceed the rank of the data matrix")
    end

    S = Xs'*Xs;

    ah = RR[:,1]
    rh = R[:,1]
    b = B[:,1]
    if n_components > 1
        vh = V[:,1]
        ph = P[:,1]
        dahds = Diagonal(I,p)
    end
    dbhds = zeros(p,p)
    drhds = zeros(p,p)
    dphds = zeros(p,p)
    dvhds = zeros(p,p)

    for i=1:n_components

        if i==1

            drhds = (Diagonal(I,p)/sqrt(ah'*S*ah))
            drhds -= ah*ah'*Xs'*(Xs)/(sqrt(ah'*S*ah)^3)
            dphds=S*drhds;
            dvhds=dphds;
            dbhds = (kron(rh,s')+rh'*s*(Diagonal(I,p)))*drhds+rh*rh'

        else

            dahds -= vh*vh'/(vh'*vh)*dahds
            dahds -= ((ah'*vh*Diagonal(I,p)+kron(ah',vh))/(vh'*vh)-2*ah'*vh*vh*vh'/(vh'*vh)^2)*dvhds;

            ah = RR[:,i]
            rh = R[:,i]

            drhds = dahds/sqrt(ah'*S*ah)-ah*ah'*Xs'*(Xs*dahds)/(sqrt(ah'*S*ah)^3);

            dphds = S*drhds
            ph=P[:,i]

            factor =  2*(ph'*vh*vh*vh')/((vh'*vh)^2)
            factor -= (ph'*vh*Diagonal(I,p)+kron(ph',vh))/(vh'*vh)
            dvhds = factor*dvhds
            dvhds += dphds - vh*vh'/(vh'*vh)*dphds

            vh = V[:,i]
            b = B[:,i]

            dbhds += (kron(rh,s')+rh'*s*(Diagonal(I,p)))*drhds + rh*rh'

        end;
    end;

    return dbhds

end;

function _fit_dbdX(n_components,n,p,Xs,ys,P,R,RR,T,V,B)

    """
    _fit_dbdX
    the univariate SIMPLS Jacobian matrices with respect to
    the predictor data
    """

    n,p = size(Xs);

    if n_components>min(n,p)
        throw("Max # LV cannot exceed the rank of the data matrix")
    end

    S = Xs'*Xs;
    s = Xs'*ys

    ah = RR[:,1]
    rh = R[:,1]
    b = B[:,1]
    dsdX = kron(Diagonal(I,p),ys')
    if n_components > 1
        vh = V[:,1]
        ph = P[:,1]
        dahdX = kron(Diagonal(I,p),ys')
    end
    dbhdX = zeros(p,n*p)
    drhdX = zeros(p,n*p)
    dphdX = zeros(p,n*p)
    dvhdX = zeros(p,n*p)

    for i=1:n_components

        if i==1

            drhdX = (Diagonal(I,p)/sqrt(ah'*S*ah))*dsdX
            drhdX -= ah*ah'*Xs'*(Xs*dsdX .+ kron(ah',Diagonal(I,n)))/(sqrt(ah'*S*ah)^3)
            dphdX = S*drhdX + kron(rh',Xs') + kron(Diagonal(I,p),rh'*Xs')
            dvhdX = dphdX
            dbhdX = (kron(rh,s')+rh'*s*(Diagonal(I,p)))*drhdX + rh*rh'*dsdX

        else

            dahdX -= vh*vh'/(vh'*vh)*dahdX
            dahdX -= (ah'*vh*Diagonal(I,p)+kron(ah',vh))*dvhdX/(vh'*vh)
            dahdX += 2*ah'*vh*vh*vh'/(vh'*vh)^2*dvhdX

            ah = RR[:,i]
            rh = R[:,i]

            drhdX = dahdX/sqrt(ah'*S*ah)
            drhdX -= ah*ah'*Xs'*(Xs*dahdX + kron(ah',Diagonal(I,n)))/(sqrt(ah'*S*ah)^3);

            dphdX = S*drhdX + kron(rh',Xs') + kron(Diagonal(I,p),rh'*Xs')
            ph=P[:,i]

            factor =  2*(ph'*vh*vh*vh')/((vh'*vh)^2)
            factor -= (ph'*vh*Diagonal(I,p)+kron(ph',vh))/(vh'*vh)
            dvhdX = factor*dvhdX
            dvhdX += dphdX - vh*vh'/(vh'*vh)*dphdX

            vh = V[:,i]
            b = B[:,i]

            dbhdX += (kron(rh,s')+rh'*s*(Diagonal(I,p)))*drhdX + rh*rh'*dsdX

        end;
    end;

    return dbhdX

end;
