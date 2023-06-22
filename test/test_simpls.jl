module TestUnisimpls

    using unisimpls
    using CSV
    using ScikitLearnBase
    using DataFrames
    using Test


    close_enough(x0, x1, tol=1e-14) = abs(x0 - x1) <= tol ? true : false

    Xf = CSV.read("../data/Xfearncal.csv", DataFrame, header=0)
    yf = CSV.read("../data/Yfearncal.csv", DataFrame, header=0)

    n,p = size(Xf)

    simreg = SIMPLS()
    set_params_dict!(simreg,Dict(:n_components => 3,:return_uncertainty => true, :centre => "mean", :scale => "std"))
    fit!(simreg,Xf,yf[:,1])

    @testset "regression coeffs and prediction" begin
        @test close_enough(length(simreg.coef_),p)
        yp = predict(simreg,Xf)
        @test all(map(close_enough, yp, simreg.fitted_))
    end

    @testset "uncertainty and Jacobians" begin
        @test size(simreg.dbdy_) == (p,n)
        @test size(simreg.dbdX_) == (p,n*p)
        pin = predint(simreg,Xf)
        @test all(map(close_enough, pin, simreg.fit_pi_))
    end
end
