using LinearAlgebra
using PlotlyJS
using KrylovKit
using ArgParse
using Dates

include("hubbard_core.jl")

"""Helper function to parse arguments from commandline."""
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--chem_max"
            help = "maximum and minimum chemical potential in range"
            default = 5.0
            arg_type = Float64
        "--chem_step"
            help = "chemical potential step"
            default = 0.1
            arg_type = Float64
        "--lattice_dims"
            help = "specify lattice sizes as d1,d2,...,dn"
            default = "2"
    end
    return parse_args(s)
end


"""Helper function to draw graphs of data and save them with a given name."""
function save_graph(chem_potentials, y_vals, U, t, temp_range, name)
    layout = Layout(
    title="$name od μ, U=$U, t=$t",
    xaxis_title="μ",
    yaxis_title="",
    )
    pl = plot([scatter(x=chem_potentials, y=yval, name="T=$t") for (yval, t) in zip(y_vals, temp_range)], layout)
    savefig(pl, "$name plot.png")
    savefig(pl, "$name plot.html")
end


function compute_thermodynamic_quantities(params, chem_pot, temperature, U, t)
    k = 1.0
    N = reduce(*, params.dim_sizes)
    H_matrix = get_hamiltonian(params, U, t, chem_pot)
    out = Array{Float64}(undef, 4N^2)
    fill!(out, NaN)
    energies, H_states, conv_info = out, out, out
    try
        energies, H_states, conv_info = eigsolve(H_matrix, 4, :SR, tol=1e-20)
    catch e
        print("Error $e at μ=$chem_pot, T=$temperature, U=$U, t=$t")
        return out, out, out, out
    end
    exp_vec = exp.(-real(energies) / (k*temperature))
    stat_sum = reduce(+, exp_vec)
    avg_particles = [no_particles_from_state(H_state, N) for H_state in H_states]
    
    densities = exp_vec ./ stat_sum
    avg_particles_temp = reduce(+, densities .* avg_particles)
    entropy = -k * reduce(+, [abs(p)*log(abs(p)) for p in densities])
    grand_potential = -k*temperature*log(stat_sum)
    return densities, avg_particles_temp, entropy, grand_potential
end


function main()
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Begin program.")

    parsed_args = parse_commandline()
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Parsed args.")

    CHEM_MAX = parsed_args["chem_max"]
    CHEM_STEP = parsed_args["chem_step"]
    STR_LATTICE_DIMS = parsed_args["lattice_dims"]
    LATTICE_DIMS = Tuple(parse.(Int64, split(STR_LATTICE_DIMS, ",")))

    N = reduce(*, LATTICE_DIMS)
    all_states = reshape(Iterators.product([[0,1] for _ in 1:2N]...) .|> collect, (1, 4^N))
    next(n) = [collect(2:n); 1]
    params = Params(LATTICE_DIMS, 
        all_states,
        [flatstate_to_fullstate(state, LATTICE_DIMS) for state in all_states],
        [next(d) for d in LATTICE_DIMS])

    chem_potentials = range(-CHEM_MAX, stop=CHEM_MAX, step=CHEM_STEP)
    temp_range = round.(10.0.^(range(-0.1,stop=1.0,length=3)), digits=3)

    U = 1.0
    t = 1.0

    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Begin calculations...")

    thermodynamics = [[compute_thermodynamic_quantities(params, chem_pot, temperature, U, t)
            for chem_pot in chem_potentials] for temperature in temp_range]

    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Calculations done.")

    avg_n = [[thermodynamics[j][i][2] for i in eachindex(chem_potentials)] for j in eachindex(temp_range)]
    entropy_vals = [[thermodynamics[j][i][3] for i in eachindex(chem_potentials)] for j in eachindex(temp_range)]
    grand_pots = [[thermodynamics[j][i][4] for i in eachindex(chem_potentials)] for j in eachindex(temp_range)]

    helmholtz_vals = [[grand_pot + chem_pot*avg_no for (grand_pot, chem_pot, avg_no) in zip(grand_pot_rows, chem_potentials, avg_no_rows)] 
        for (grand_pot_rows, avg_no_rows) in zip(grand_pots, avg_n)]

    U_vals = [[helm + entr*temp for (helm, entr) in zip(helmholtz_vals_rows, entr_rows)]
        for (temp, helmholtz_vals_rows, entr_rows) in zip(temp_range, helmholtz_vals, entropy_vals)]

    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Saving plots...")
    save_graph(chem_potentials, avg_n, U, t, temp_range, "$STR_LATTICE_DIMS srednia liczba czastek")
    save_graph(chem_potentials, entropy_vals, U, t, temp_range, "$STR_LATTICE_DIMS entropia")
    save_graph(chem_potentials, grand_pots, U, t, temp_range, "$STR_LATTICE_DIMS wielki potencjal")
    save_graph(chem_potentials, helmholtz_vals, U, t, temp_range, "$STR_LATTICE_DIMS energia sw helmholtza")
    save_graph(chem_potentials, U_vals, U, t, temp_range, "$STR_LATTICE_DIMS energia wewnetrzna")
    
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Finished.")
end

main()