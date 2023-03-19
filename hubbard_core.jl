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
    return out
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
    return out
end

up(cell_no) = Int(2cell_no - 1)
down(cell_no) = Int(2cell_no)

state_to_idx(state) = reduce(+, state .* [2^n for n in 0:length(state)-1]) + 1
state_to_statevec(state) = [x == state_to_idx(state) ? 1.0 : 0.0 for x in 1:(length(state))^2]

# H |state>
interacting_term(state, U, N) = U * reduce(+, [state[up(n)] * state[down(n)] for n in 1:N]) .* state_to_statevec(state)

function kinetic_term(state, t, next) 
    N = div(length(state), 2)
    total = zeros((2N)^2)
    for n in 1:(N)
        tmp1 = annihilate(state, up(next[n]))
        tmp2 = create(tmp1, up(n))
        if typeof(tmp2) != Int
            total[state_to_idx(tmp2)] += 1
        end
        tmp3 = annihilate(state, down(next[n]))
        tmp4 = create(tmp3, down(n))
        if typeof(tmp4) != Int
            total[state_to_idx(tmp4)] += 1
        end
    end
    return t * total
end

# perform H |state> and get statevec of: alpha |state'>
function hamiltonian_on_state(U, t, chem_pot, state, next)
    N = div(length(state), 2)
    # @show state
    interacting = interacting_term(state, U, N)
    # @show interacting
    kinetic = kinetic_term(state, t, next)
    # @show kinetic
    chemical = chem_pot * reduce(+, state) .* state_to_statevec(state)
    interacting + kinetic - chemical
end


normalize_statevec(statevec) = [x == 0 ? 0 : 1 for x in statevec]


function braket(bra, ket)
#     if normalize_statevec(bra) == normalize_statevec(ket)
        return dot(bra, ket)
#     else
#         return 0.0
#     end
end


# for every H |state> multiply it by every possible <state'|
function get_hamiltonian(states, U, t, chem_pot, next, N)
    hamiltonian = zeros((2N)^2, (2N)^2)
    right_side = [hamiltonian_on_state(U, t, chem_pot, state, next) for state in states]
    for i in 1:(2N)^2, j in 1:(2N)^2
        hamiltonian[i, j] = braket(state_to_statevec(states[i]), right_side[j])
    end
    hamiltonian
end


function no_particles_from_state(state, N)
    multiplier = [reduce(+, digits(n, base=2)) for n in 0:((2N)^2 -1)]
    dot(state, multiplier.*state)
end