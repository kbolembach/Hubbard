__precompile__()

using SparseArrays
using Combinatorics

struct Params
    dim_sizes # tuple with sizes of each dimension
    next
end


struct ThermodynamicsResult
    energies
    states
    density
    avg_number_particles
    entropy
    grand_potential
end


function move_particle(state_vector, spin, original_pos, target_pos)
    if typeof(state_vector) == Int
        return -1
    end
    if state_vector[original_pos...][spin] == 0 || state_vector[target_pos...][spin] == 1
        return -1
    end

    state_vector[original_pos...][spin]  = 0
    state_vector[target_pos...][spin]  = 1
    idx = flatstate_to_idx(state_vector)
    # undo changes:
    state_vector[original_pos...][spin]  = 1
    state_vector[target_pos...][spin]  = 0

    return idx
end

# get the tuples of coordinates for each neighbor of idx
function forward_neighbors(idx, next)
    out = []
    t_idx = [i for i in Tuple(idx)]
    for i in eachindex(t_idx)
        neighbor = copy(t_idx)
        neighbor[i] = next[i][t_idx[i]]
        if Tuple(neighbor) != Tuple(t_idx)
            push!(out, Tuple(neighbor))
        end
    end
    out
end

function idx_to_flatstate(idx, N::Number)
    if idx > 4^N - 1
        throw(DomainError(idx, "Index larger than possible state."))
    end
    out = digits(idx, base=2)
    while length(out) != 2N
        append!(out, 0)
    end

    return out
end


function idx_to_flatstate(idx, dim_sizes::Tuple{Vararg{Int}})
    N = reduce(*, dim_sizes)
    idx_to_flatstate(idx, N)
end


function idx_to_fullstate(idx, dim_sizes)
    flatstate_to_fullstate(idx_to_flatstate(idx, dim_sizes), dim_sizes)
end

# Converts bit state vector to the index it represents (e.g. [1, 1, 0, 0] -> 4)
function flatstate_to_idx(state) 
    state = collect(Iterators.flatten(state))
    reduce(+, state .* [2^n for n in 0:length(state)-1]) + 1
end

function fullstate_to_idx(fullstate)
    state = collect(Iterators.flatten(fullstate)) 
    flatstate_to_idx(state)
end

# Converts bit state vector to the full state vector (e.g. [1, 1, 0, 0] becomes a vector of length 16)
function state_to_statevec(state) 
    state = collect(Iterators.flatten(state))   
    # [x == state_to_idx(state) ? 1.0 : 0.0 for x in 1:2^(length(state))] 
    out = zeros(2^(length(state)))
    out[flatstate_to_idx(state)] = 1
    return out
end

function flatstate_to_fullstate(flatstate, dim_sizes)
    halfstate = [[flatstate[i], flatstate[i+1]] for i in range(1, step=2, stop=length(flatstate))]
    reshape(halfstate, dim_sizes)
end

# perform H |state>
function hamiltonian_on_state(params::Params, U, t, chem_pot, state)
    # total = reshape([[0,0] for _ in 1:reduce(*, params.dim_sizes)], params.dim_sizes)
    N = reduce(*, params.dim_sizes)
    total = sparsevec([], [], 4^N)
    
    stvecidx = fullstate_to_idx(state)
    # (interacting - chem) term
    total[stvecidx] = U * reduce(+, [pos[1]*pos[2] for pos in state]) - chem_pot * reduce(+, collect(Iterators.flatten(state)))

    # kinetic term
    for idx in CartesianIndices(state)
        for neighbor in forward_neighbors(idx, params.next)
            idx1 = move_particle(state, 1, neighbor, Tuple(idx))
            if idx1 != -1
                total[idx1] = t
            end

            idx2 = move_particle(state, 2, neighbor, Tuple(idx))
            if idx2 != -1
                total[idx2] = t
            end
        end
    end

    return total
end


function hamiltonian_on_state(params::Params, U, t, chem_pot, 
    fullstate, no_electrons, inv_ids)

    # total = reshape([[0,0] for _ in 1:reduce(*, params.dim_sizes)], params.dim_sizes)
    N = reduce(*, params.dim_sizes)
    len_flatstates = binomial(2N, no_electrons)
    # base_flatstate = [ones(Int, no_electrons)..., zeros(Int, 2N - no_electrons)...]
    # flatstates = multiset_permutations(base_flatstate, 2N)
    # indices = flatstate_to_idx.(flatstates)
    
    total = sparsevec([], [], len_flatstates)

    # function fullidx_to_limitedidx(full_idx, indices)
    #     findfirst(isequal(full_idx), indices)
    # end
    stvecidx = inv_ids[fullstate_to_idx(fullstate)]
    # stvecidx = findfirst(isequal(flatstate), flatstates)

    # (interacting - chem) term
    total[stvecidx] = U * reduce(+, [pos[1]*pos[2] for pos in fullstate]) - chem_pot * reduce(+, collect(Iterators.flatten(fullstate)))

    # kinetic term
    for idx in CartesianIndices(fullstate)
        for neighbor in forward_neighbors(idx, params.next)
            for spin in [1, 2]
                idx1 = move_particle(fullstate, spin, neighbor, Tuple(idx))
                if idx1 != -1
                    total[inv_ids[idx1]] = t
                end
            end
        end
    end

    return total
end


function braket(bra, ket)
    dot(bra, ket)
end


# for every H |state> multiply it by every possible <state'|
function get_hamiltonian(hamiltonian::SparseMatrixCSC, params::Params, 
    U, t, chem_pot)

    N = reduce(*, params.dim_sizes)
    # hamiltonian = zeros(4^N, 4^N)
    # hamiltonian = sprand(4^N, 4^N, 16.0^-N) ./ 10^6

    fn(idx) = hamiltonian_on_state(params, U, t, chem_pot, idx_to_fullstate(idx, params.dim_sizes)) # for idx in 0:(4^N - 1)
    for j in 1:4^N
        # left_state = [i == idx ? 1 : 0 for idx in 1:4^N]
        # hamiltonian[i, j] = braket(left_state, right_side[j])
        idxs, vals = findnz(fn(j-1))
        for (i, val) in zip(idxs, vals)
            if val != 0
                hamiltonian[i, j] = val
                if i != j
                    hamiltonian[j, i] = conj(val)
                end
            end
        end
    end
    
    return hamiltonian
end

# get the hamiltonian for a set number of electrons
function get_hamiltonian(params::Params, U, t, chem_pot, no_e, sparse, ids, inv_ids)

    N = reduce(*, params.dim_sizes)
    len_flatstates = binomial(2N, no_e)
    
    if sparse
        hamiltonian = sprand(len_flatstates, len_flatstates, len_flatstates^(-2.0)) ./ 10^6
    else
        hamiltonian = zeros(len_flatstates, len_flatstates)
    end

    for (j, id) in enumerate(ids) 
        idxs, vals = findnz(hamiltonian_on_state(params, U, t, chem_pot, idx_to_fullstate(id-1, params.dim_sizes), no_e, inv_ids))
        for (i, val) in zip(idxs, vals)
            if val != 0
                hamiltonian[i, j] = val
                if i != j
                    hamiltonian[j, i] = conj(val)
                end
            end
        end
    end
    
    return hamiltonian
end


function get_hamiltonian(hamiltonian::Matrix, params::Params, U, t, chem_pot)

    N = reduce(*, params.dim_sizes)

    fn(idx) = hamiltonian_on_state(params, U, t, chem_pot, idx_to_fullstate(idx, params.dim_sizes)) # for idx in 0:(4^N - 1)
    for j in 1:4^N
        idxs, vals = findnz(fn(j-1))
        for (i, val) in zip(idxs, vals)
            if val != 0
                hamiltonian[i, j] = val
                if i != j
                    hamiltonian[j, i] = conj(val)
                end
            end
        end
    end
    
    return hamiltonian
end


function no_particles_from_state(state, multiplier)
    # include this multiplier in general code for better performance:
    # multiplier = [reduce(+, digits(n, base=2)) for n in 0:(4^N -1)]
    braket(state, multiplier.*state)
end

# z symetrią liczby cząstek
function compute_thermodynamic_quantities(params, chem_pot, temperature, U, t, avg_no_multiplier)
    
    N = reduce(*, params.dim_sizes)
    k = 1.0

    energies = []
    H_states = []
    avg_particles = []
    for no_e in 0:2N
        base_flatstate = [ones(Int, no_e)..., zeros(Int, 2N - no_e)...]
        flatstates = multiset_permutations(base_flatstate, 2N)
        no_possible_states = binomial(2N, no_e)
        ids = [flatstate_to_idx(state) for state in flatstates]
        inv_ids = Dict(id => i for (id, i) in zip(ids, 1:no_possible_states))

        if no_possible_states > 100
            H_matrix = get_hamiltonian(params, U, t, chem_pot, no_e, true, ids, inv_ids)
            out = Array{Float64}(undef, no_possible_states)
            fill!(out, NaN)
            t_energies, t_H_states, conv_info = out, out, out
            try
                t_energies, t_H_states, conv_info = eigsolve(H_matrix, 4, :SR, tol=1e-27)
                append!(energies, t_energies)
                append!(H_states, t_H_states)
                append!(avg_particles, [real(dot(state, no_e.*state)) for state in t_H_states]...)
            catch e
                println("Error $e at μ=$chem_pot, T=$temperature, U=$U, t=$t")
                return out, NaN, NaN, NaN
            end
        else
            # hamiltonian = zeros(no_possible_states, no_possible_states)
            H_matrix = get_hamiltonian(params, U, t, chem_pot, no_e, false, ids, inv_ids)
            diagonalized = eigen(H_matrix)
            t_energies = diagonalized.values
            t_H_states = Vector{Float64}[eachcol(diagonalized.vectors)...]
            append!(energies, float.(t_energies))
            append!(H_states, t_H_states)
            append!(avg_particles, [real(dot(t_state, no_e.*t_state)) for t_state in t_H_states]...)
        end
    end
    exp_vec = exp.(-real(energies) / (k*temperature))
    stat_sum = reduce(+, exp_vec)
    # avg_particles = [no_particles_from_state(H_state, avg_no_multiplier) for H_state in H_states]
    
    densities = real(exp_vec ./ stat_sum)
    avg_particles_temp = reduce(+, densities .* avg_particles)
    entropy = -k * reduce(+, [abs(p)*log(abs(p)) for p in densities])
    grand_potential = -k*temperature*log(stat_sum)

    return ThermodynamicsResult(energies, H_states, densities, avg_particles_temp, 
        entropy, grand_potential)
end

# tylko lanczos
# function compute_thermodynamic_quantities(params, chem_pot, temperature, U, t, avg_no_multiplier)
#     N = reduce(*, params.dim_sizes)
#     hamiltonian = sprand(4^N, 4^N, 16.0^-N) ./ 10^6
#     H_matrix = get_hamiltonian(hamiltonian, params, U, t, chem_pot)
#     out = Array{Float64}(undef, 4N^2)
#     fill!(out, NaN)
#     energies, H_states, conv_info = out, out, out
#     try
#         energies, H_states, conv_info = eigsolve(H_matrix, 4, :SR, tol=1e-27)
#     catch e
#         print("Error $e at μ=$chem_pot, T=$temperature, U=$U, t=$t")
#         return out, out, out, out
#     end
#     exp_vec = exp.(-real(energies) / (k*temperature))
#     stat_sum = reduce(+, exp_vec)
#     avg_particles = [no_particles_from_state(H_state, avg_no_multiplier) for H_state in H_states]
    
#     densities = exp_vec ./ stat_sum
#     avg_particles_temp = reduce(+, densities .* avg_particles)
#     entropy = -k * reduce(+, [abs(p)*log(abs(p)) for p in densities])
#     grand_potential = -k*temperature*log(stat_sum)
#     return densities, avg_particles_temp, entropy, grand_potential
# end