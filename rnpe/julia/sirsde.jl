# Contains script to run SIRSDE model using command line.

using LinearAlgebra
using StochasticDiffEq
using Base: @kwdef

using Random
using NPZ
using ArgParse


# Notation
# s: Susceptible
# i: Infected
# r: Removed/recovered/dead
# β: Infection rate
# γ: recovery rate 
# R₀: infection_rate/recovery rate  (i.e. β/γ)
# σ: Controls volitility of R₀
# η: Controls mean reversion strength of R₀ 


"""
SIR model described using stochastic differential equations.
Ref: Inspired by https://julia.quantecon.org/continuous_time/covid_sde.html
"""
@kwdef struct SIRSDETask
    "Volatitility of R₀."
    σ::Float64 = 0.05
    "Mean reversion strength of R₀."
    η::Float64 = 0.05
    "Number of time steps."
    T::Int = 365
    "Number of individuals"
    N::Int = 100000
    "Initial proportions of [s,i,r]."
    sir_0::Vector{Float64} = [0.999, 0.001, 0]
    "Deterministic portion of SDE."
    f::Function = _sir_f!
    "Stochastic portion of SDE."
    g::Function = _sir_g!
    "Parameter names."
    θ_names::Vector{String} = ["β", "γ"]
    name::String = "SIRSDE"
end

"""
Deterministic portion of SDE (s, i, r are described using proportions).
"""
function _sir_f!(du, u, p, t)
    s, i, r, R₀ = u
    (; γ, R̄₀, η, σ) = p
    β = γ * R₀
    du[1] = -β * s * i        # ds/dt
    du[2] = β * s * i - γ * i   # di/dt
    du[3] = γ * i           # dr/dt
    du[4] = η * (R̄₀(t, p) - R₀) # dR₀/dt
    return nothing
end

"""
Stochastic portion of SDE (R₀ volatitility)
"""
function _sir_g!(du, u, p, t)
    s, i, r, R₀ = u
    (; γ, R̄₀, η, σ) = p
    du[1:3] .= 0
    du[4] = σ * sqrt(abs(R₀))
    return nothing
end


"""
Batched simulations using SIR model. θ has columns [β γ].
"""
function simulate(
    task::SIRSDETask,
    θ::AbstractMatrix{Float64})
    @assert all(θ .>= 0)
    (; σ, η, T, sir_0, f, g, N) = task

    x = Array{Float64}(undef, size(θ, 1), T)  # To store infection data
    for (θᵢ, xᵢ) in zip(eachrow(θ), eachrow(x))
        β, γ = θᵢ
        p = (
            R̄₀_ref=β / γ,  # Fixed reference (R̄₀ must be a function)
            R̄₀=(t, p) -> p.R̄₀_ref,
            γ=γ,
            σ=σ,
            η=η,
        )
        u_0 = [sir_0; β / γ]
        prob = SDEProblem(f, g, u_0, (0, T), p)
        sol = solve(prob, SOSRI())
        xᵢ .= [sol(t)[2] for t in 0:(T-1)]
    end
    return x .* N
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--seed"
        help = "Seed for simulations"
        arg_type = Int
        required = true

        "--theta_path"
        help = "Path to prior samples (npz file) to use for simulations"
        arg_type = String
        required = true

        "--output_path"
        help = "Path to output raw simulations"
        arg_type = String
        required = true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    Random.seed!(args["seed"])
    θ = npzread(args["theta_path"])["theta"]
    θ = convert(Matrix{Float64}, θ)
    task = SIRSDETask()
    x = simulate(task, θ)
    npzwrite(args["output_path"], x=x)
end

main()

