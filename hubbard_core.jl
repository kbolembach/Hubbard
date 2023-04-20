struct Params
    dim_sizes # tuple with sizes of each dimension
    next
end

""" 
    create(state_vector, pos)

Acts with a creation operator upon a state.


### Input

 - `state_vector` -- a vector with information on the state in the bit representation, i.e. each element is either 1 or 0 depending if there is a boson in the given position. 
 - `pos` -- an integer representing the position; for models including spin, odd positions (2n - 1) are spin up, even positions (2n) are spin down, n being the lattice point index.
 
### Output

A modified copy of the state vector, or scalar 0 if the target position was already occupied or the state vector was an integer (for redundancy reasons).

### Examples

```julia-repl
julia> create([0, 0, 1, 0], 2)
4-element Vector{Int64}:
 0
 1
 1
 0
 ```"""
function create(state_vector, pos)
    if typeof(state_vector) == Int
        return 0
    end
    if state_vector[pos] == 1
        return 0
    end
    out = copy(state_vector)
    out[pos] = 1
    
    out
end


""" 
    annihilate(state_vector, pos)

Acts with an annihilation operator upon a state.


### Input

 - `state_vector` -- a vector with information on the state in the bit representation, i.e. each element is either 1 or 0 depending if there is a boson in the given position. 
 - `pos` -- an integer representing the position; for models including spin, odd positions (2n - 1) are spin up, even positions (2n) are spin down, n being the lattice point index.
 
### Output

A modified copy of the state vector, or scalar 0 if the target position was already occupied or the state vector was an integer (for redundancy reasons).

### Examples

```julia-repl
julia> annihilate([0, 1, 1, 0], 3)
4-element Vector{Int64}:
 0
 1
 0
 0
 ```"""
function annihilate(state_vector, pos)
    if typeof(state_vector) == Int
        return 0
    end
    if state_vector[pos] == 0
        return 0
    end
    out = copy(state_vector)
    out[pos] = 0
    
    out
end

function move_particle(state_vector, spin, original_pos, target_pos)
    if typeof(state_vector) == Int
        return 0
    end
    if state_vector[original_pos...][spin] == 0 || state_vector[target_pos...][spin] == 1
        return 0
    end

    out = deepcopy(state_vector)
    out[original_pos...][spin]  = 0
    out[target_pos...][spin]  = 1

    return out
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
function state_to_idx(state) 
    state = collect(Iterators.flatten(state))
    reduce(+, state .* [2^n for n in 0:length(state)-1]) + 1
end

# Converts bit state vector to the full state vector (e.g. [1, 1, 0, 0] becomes a vector of length 16)
function state_to_statevec(state) 
    state = collect(Iterators.flatten(state))   
    [x == state_to_idx(state) ? 1.0 : 0.0 for x in 1:2^(length(state))]
end

function flatstate_to_fullstate(flatstate, dim_sizes)
    halfstate = [[flatstate[i], flatstate[i+1]] for i in range(1, step=2, stop=length(flatstate))]
    reshape(halfstate, dim_sizes)
end

# perform H |state> and get statevec of: alpha |state'>
function hamiltonian_on_state(params::Params, U, t, chem_pot, state)
    # total = reshape([[0,0] for _ in 1:reduce(*, params.dim_sizes)], params.dim_sizes)
    N = reduce(*, params.dim_sizes)
    total = zeros(2^(2N))

    # nonzero_entries_indices = [] # przepisać kod, by skipował zera
    for idx in CartesianIndices(state)
        for neighbor in forward_neighbors(idx, params.next)
            tmp1 = move_particle(state, 1, neighbor, Tuple(idx))
            if typeof(tmp1) != Int
                total[state_to_idx(tmp1)] += 1
            end

            tmp2 = move_particle(state, 2, neighbor, Tuple(idx))
            if typeof(tmp2) != Int
                total[state_to_idx(tmp2)] += 1
            end
        end
    end

    kinetic = t * total
    interacting = U * reduce(+, [pos[1]*pos[2] for pos in state]) .* state_to_statevec(state)
    chemical = chem_pot * reduce(+, collect(Iterators.flatten(state))) .* state_to_statevec(state)
    
    interacting + kinetic - chemical
end


function braket(bra, ket)
    dot(bra, ket)
end


# for every H |state> multiply it by every possible <state'|
function get_hamiltonian(params::Params, U, t, chem_pot)
    N = reduce(*, params.dim_sizes)
    hamiltonian = zeros(4N^2, 4N^2)

    right_side = [hamiltonian_on_state(params, U, t, chem_pot, idx_to_fullstate(idx, params.dim_sizes)) for idx in 0:(4^N - 1)]
    for i in 1:4N^2, j in 1:4N^2
        left_state = [i == idx ? 1 : 0 for idx in 1:4^N]
        hamiltonian[i, j] = braket(left_state, right_side[j])
    end
    
    hamiltonian
end


function no_particles_from_state(state, N)
    multiplier = [reduce(+, digits(n, base=2)) for n in 0:(4N^2 -1)]
    braket(state, multiplier.*state)
end