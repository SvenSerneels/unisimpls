function _check_is_fitted(self;method_name::Symbol=:spls)

    """
    Check if SIMPLS model has been fit
    """

    if self.coef_ == nothing
        return false
    else
        return true
    end
end #_check_is_fitted

function _predict_check(self,Xn)

    """
    Check data and SIMPLS object prior to making makings and change formats
    """

    @assert _check_is_fitted(self) "Please fit model before making predictions"
    Xn_type = typeof(Xn)
    @assert Xn_type in vcat(self.X_Types_Accept,self.y_Types_Accept) "Supply new X data as DataFrame or Matrix"
    if Xn_type == DataFrame
        Xn = Matrix(Xn)
    end
    if Xn_type in [DataFrameRow,DataFrameRow{DataFrames.DataFrame,DataFrames.Index}]
        Xn = Array(Xn)
    end
    nxn = size(Xn)
    if length(nxn)==1 #Transform one new case
        Xn = reshape(Xn,(1,nxn[1]))
        pxn = nxn[1]
        nxn = 1
    else
        pxn = nxn[2]
        nxn = nxn[1]
    end

    return(Xn,nxn,pxn)

end #_predict_check
