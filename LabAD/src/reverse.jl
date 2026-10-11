module Reverse

import ComputationGraphExplorer as CGE
import ..Forward

mutable struct ReverseData{T}
    derivative::T
    local_jac::Vector{T}
end

CGE.metadata(::Type{ReverseData{T}}, ::T) where {T} = ReverseData{T}(zero(T), T[])
CGE.metadata_rows(data::ReverseData) = [
    "r" => data.derivative,
    "J" => data.local_jac,
]

const RevNode{T} = CGE.Node{T,ReverseData{T}}
const Node = RevNode{Float64}
const DualNode = RevNode{Forward.Dual{Float64}}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? one(data.derivative) : zero(data.derivative)
end

to_scalar(::Type{T}, value) where {T} = value isa T ? value : T(value)
to_scalar(::Type{Forward.Dual{S}}, value) where {S} =
    value isa Forward.Dual ? value : Forward.Dual{S}(S(value), zero(S))

function new_node(op, value, args::Vector{RevNode{T}}, local_jac) where {T}
    jac = map(local_jac) do coefficient
        to_scalar(T, coefficient)
    end
    return RevNode{T}(op, args, value, ReverseData{T}(zero(T), jac))
end

Base.zero(::RevNode{T}) where {T} = RevNode{T}(zero(T))
Base.:*(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:*, x.value * y.value, RevNode{T}[x, y], [y.value, x.value])
Base.:*(x::RevNode{T}, y::Number) where {T} = new_node(:*, x.value * y, RevNode{T}[x], [y])
Base.:*(x::Number, y::RevNode{T}) where {T} = new_node(:*, x * y.value, RevNode{T}[y], [x])
Base.:+(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:+, x.value + y.value, RevNode{T}[x, y], [1, 1])
Base.:+(x::RevNode{T}, y::Number) where {T} = new_node(:+, x.value + y, RevNode{T}[x], [1])
Base.:+(x::Number, y::RevNode{T}) where {T} = new_node(:+, x + y.value, RevNode{T}[y], [1])
Base.:-(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:-, x.value - y.value, RevNode{T}[x, y], [1, -1])
Base.:-(x::RevNode{T}, y::Number) where {T} = new_node(:-, x.value - y, RevNode{T}[x], [1])
Base.:-(x::Number, y::RevNode{T}) where {T} = new_node(:-, x - y.value, RevNode{T}[y], [-1])
Base.:-(x::RevNode{T}) where {T} = new_node(:-, -x.value, RevNode{T}[x], [-1])
Base.:/(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:/, x.value / y.value, RevNode{T}[x, y], [1 / y.value, -x.value / y.value^2])
Base.:/(x::RevNode{T}, y::Number) where {T} = new_node(:/, x.value / y, RevNode{T}[x], [1 / y])
Base.:/(x::Number, y::RevNode{T}) where {T} = new_node(:/, x / y.value, RevNode{T}[y], [-x / y.value^2])
Base.:^(x::RevNode{T}, n::Integer) where {T} = Base.power_by_squaring(x, n)
Base.tanh(x::RevNode{T}) where {T} = new_node(:tanh, tanh(x.value), RevNode{T}[x], [1 - tanh(x.value)^2])
Base.exp(x::RevNode{T}) where {T} = new_node(:exp, exp(x.value), RevNode{T}[x], [exp(x.value)])
Base.log(x::RevNode{T}) where {T} = new_node(:log, log(x.value), RevNode{T}[x], [1 / x.value])

function CGE.pullback!(f::RevNode{T}) where {T}
    for (parent, local_jac) in zip(f.args, f.metadata.local_jac)
        parent.metadata.derivative += f.metadata.derivative * local_jac
    end
end

"""Only supports scalar valued functions."""
function gradient!(f, g, x)
    x_nodes = map(Node, x)
    root = f(x_nodes)
    CGE.backward!(root)
    return map!(node -> node.metadata.derivative, g, x_nodes)
end

"""Only supports scalar valued functions."""
gradient(f, x) = gradient!(f, zero(x), x)

# --- jacobian and hessian --- #
export gradient, jacobian, hessian

"""Only supports vector valued functions."""
function jacobian(f, x)
    # J_ij = df_i/dx_j
    inputs = map(Node, x)
    outputs = f(inputs)
    jac = zeros(eltype(x), length(outputs), length(x))
    for (i, output) in enumerate(outputs)
        # Reset the derivatives for the next output
        for input in inputs
            input.metadata.derivative = 0.0
        end

        CGE.backward!(output)
        jac[i, :] = map(node -> node.metadata.derivative, inputs)
    end
    return jac
end

"""Only supports scalar valued functions."""
function hessian(f, x)
    hess = zeros(eltype(x), length(x), length(x))
    for j in eachindex(x)
        direction = zeros(eltype(x), length(x))
        direction[j] = 1
        inputs = map(Forward.Dual, x, direction)
        nodes = map(DualNode, inputs)
        output = f(nodes)
        CGE.backward!(output)
        hess[:, j] = map(node -> node.metadata.derivative.derivative, nodes)
    end
    return hess
end

end


