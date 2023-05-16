using LinearAlgebra
using PlotlyJS
using KrylovKit
using ArgParse
using Dates
using InteractiveUtils
using DelimitedFiles

include("hubbard_core.jl")

"""Helper function to parse arguments from commandline."""
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--U"
            help = "strength of the coulomb interaction"
            default = 1.0
            arg_type = Float64
        "--t"
            help = "hopping integral"
            default = 1.0
            arg_type = Float64
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
        "--temperatures"
            help = "specify lattice sizes as t1,t2,...,tn"
            default = "1.0"
    end
    return parse_args(s)
end


"""Helper function to draw graphs of data and save them with a given name."""
function save_graph(path, chem_potentials, y_vals, U, t, temp_range, name)
    str_temp_range = string(temp_range)
    layout = Layout(
        title="$name od μ, U=$U, t=$t; temperatury $str_temp_range",
        xaxis_title="μ",
        yaxis_title="",
    )
    pl = plot([scatter(x=chem_potentials, y=yval, name="T=$t") for (yval, t) in zip(y_vals, temp_range)], layout)
    savefig(pl, path * "$name plot.png")
    savefig(pl, path * "$name plot.html")
    for (temp, data) in zip(temp_range, y_vals)
        writedlm(path * "$name, temp=$temp.csv", data)
    end
end


k = 1.0

function main()
    t_start_program = now()
    n_threads = Threads.nthreads()
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Begin program with $n_threads threads.")

    parsed_args = parse_commandline()
    CHEM_MAX = parsed_args["chem_max"]
    CHEM_STEP = parsed_args["chem_step"]
    STR_LATTICE_DIMS = parsed_args["lattice_dims"]
    LATTICE_DIMS = Tuple(parse.(Int64, split(STR_LATTICE_DIMS, ",")))
    N = reduce(*, LATTICE_DIMS)
    U = parsed_args["U"]
    t = parsed_args["t"]
    STR_TEMPERATURES = parsed_args["temperatures"]
    temp_range = Tuple(parse.(Float64, split(STR_TEMPERATURES, ",")))
    
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Parsed args.")

    nw = Dates.format(now(), "mm-dd HH-MM-SS")
    path = "./hubbard_results/$STR_LATTICE_DIMS T=$nw/"
    mkpath(path)

    chem_potentials = range(-CHEM_MAX, stop=CHEM_MAX, step=CHEM_STEP)
    multiplier = [reduce(+, digits(n, base=2)) for n in 0:(4^N -1)]
    next(n) = [collect(2:n); 1]
    params = Params(LATTICE_DIMS, [next(d) for d in LATTICE_DIMS])

    t_start_calculations = now()
    nw = Dates.format(t_start_calculations, "mm-dd HH:MM:SS")
    println("$nw  Begin calculations...")

    avg_n = Array{Array{Float64}}(undef, length(temp_range))
    entropy_vals = Array{Array{Float64}}(undef, length(temp_range))
    grand_pots = Array{Array{Float64}}(undef, length(temp_range))

    for temp_idx in eachindex(temp_range)
        temp_avg_n = Array{Float64}(undef, length(chem_potentials))
        temp_entropy_vals = Array{Float64}(undef, length(chem_potentials))
        temp_grand_pots = Array{Float64}(undef, length(chem_potentials))
        Threads.@threads for chem_idx in eachindex(chem_potentials)
            density, avg_particles, entropy, grand_pot = compute_thermodynamic_quantities(params, 
                chem_potentials[chem_idx], temp_range[temp_idx], U, t, multiplier)
            
            temp_avg_n[chem_idx] = avg_particles
            temp_entropy_vals[chem_idx] = entropy
            temp_grand_pots[chem_idx] = grand_pot
        end
        avg_n[temp_idx] = temp_avg_n
        entropy_vals[temp_idx] = temp_entropy_vals
        grand_pots[temp_idx] = temp_grand_pots
    end

    t_end_calculations = now()
    nw = Dates.format(t_end_calculations, "mm-dd HH:MM:SS")
    println("$nw  Calculations done.")

    helmholtz_vals = [[grand_pot + chem_pot*avg_no for (grand_pot, chem_pot, avg_no) in zip(grand_pot_rows, chem_potentials, avg_no_rows)] 
        for (grand_pot_rows, avg_no_rows) in zip(grand_pots, avg_n)]

    U_vals = [[helm + entr*temp for (helm, entr) in zip(helmholtz_vals_rows, entr_rows)]
        for (temp, helmholtz_vals_rows, entr_rows) in zip(temp_range, helmholtz_vals, entropy_vals)]

    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Saving plots...")
    mkpath("/$STR_LATTICE_DIMS")
    save_graph(path, chem_potentials, avg_n, U, t, temp_range, "Srednia liczba czastek")
    save_graph(path, chem_potentials, entropy_vals, U, t, temp_range, "Entropia")
    save_graph(path, chem_potentials, grand_pots, U, t, temp_range, "Wielki potencjal")
    save_graph(path, chem_potentials, helmholtz_vals, U, t, temp_range, "Energia sw helmholtza")
    save_graph(path, chem_potentials, U_vals, U, t, temp_range, "Energia wewnetrzna")

    t_end_program = now()
    nw = Dates.format(t_end_program, "mm-dd HH:MM:SS")
    total_calc_time = string(t_end_calculations - t_start_calculations)
    total_exec_time = string(t_end_program - t_start_program)
    println("$nw  Finished in $total_calc_time (total: $total_exec_time).")
end

main()