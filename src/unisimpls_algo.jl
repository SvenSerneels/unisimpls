@doc """
`unisimpls_algo` Univariate SIMPLS Algorithm

Can be called directly or though the scikit-learn API.

Written by Sven Serneels.

""" ->

function _fit_simpls(n_components,n,p,Xs,ys=[],s=[],all_components=false,verbose=false)

    if verbose
        print("Fitting a " * string(n_components) * " SIMPLS model")
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
       VV(:,a) = v
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
