module Reverse

import ComputationGraphExplorer as CGE
import ..Forward

const FloatType = Float64

mutable struct ReverseData{T}
    derivative::T
    local_jac::Vector{T}
end
ReverseData{T}(derivative::T) where {T} = ReverseData{T}(derivative, T[])

CGE.metadata(::Type{ReverseData{T}}, ::T) where {T} = ReverseData{T}(zero(T))
CGE.metadata_rows(data::ReverseData) = [
    "r" => data.derivative,
    "J" => data.local_jac,
]

const RevNode{T} = CGE.Node{T,ReverseData{T}}
const Node = RevNode{FloatType}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? one(data.derivative) : zero(data.derivative)
end

function new_node(op, value::T, args::Vector{RevNode{T}}, local_jac::Vector{T}) where {T}
    data = ReverseData{T}(zero(T), local_jac)
    return RevNode{T}(op, args, value, data)
end

function new_node(op, value, args::Vector{RevNode{T}}, local_jac::Vector{T}) where {T}
    data = ReverseData{T}(zero(T), local_jac)
    return RevNode{T}(op, args, convert(T, value), data)
end

function new_node(op, value::T, args::Vector{RevNode{T}}, local_jac) where {T}
    jac = convert.(T, local_jac)
    data = ReverseData{T}(zero(T), jac)
    return RevNode{T}(op, args, value, data)
end

Base.zero(::RevNode{T}) where {T} = RevNode{T}(zero(T))
Base.zero(::Type{RevNode{T}}) where {T} = RevNode{T}(zero(T))
Base.one(::RevNode{T}) where {T} = RevNode{T}(one(T))
Base.one(::Type{RevNode{T}}) where {T} = RevNode{T}(one(T))

Base.:*(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:*, x.value * y.value, RevNode{T}[x, y], [y.value, x.value])
Base.:*(x::RevNode{T}, y::Number) where {T} = new_node(:*, x.value * y, RevNode{T}[x], [y])
Base.:*(x::Number, y::RevNode{T}) where {T} = new_node(:*, x * y.value, RevNode{T}[y], [x])
Base.:+(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:+, x.value + y.value, RevNode{T}[x, y], [one(T), one(T)])
Base.:+(x::RevNode{T}, y::Number) where {T} = new_node(:+, x.value + y, RevNode{T}[x], [one(T)])
Base.:+(x::Number, y::RevNode{T}) where {T} = new_node(:+, x + y.value, RevNode{T}[y], [one(T)])
Base.:-(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:-, x.value - y.value, RevNode{T}[x, y], [one(T), -one(T)])
Base.:-(x::RevNode{T}, y::Number) where {T} = new_node(:-, x.value - y, RevNode{T}[x], [one(T)])
Base.:-(x::Number, y::RevNode{T}) where {T} = new_node(:-, x - y.value, RevNode{T}[y], [-one(T)])
Base.:-(x::RevNode{T}) where {T} = new_node(:-, -x.value, RevNode{T}[x], [-one(T)])
Base.:/(x::RevNode{T}, y::RevNode{T}) where {T} = new_node(:/, x.value / y.value, RevNode{T}[x, y], [one(T) / y.value, -x.value / y.value^2])
Base.:/(x::RevNode{T}, y::Number) where {T} = new_node(:/, x.value / y, RevNode{T}[x], [one(T) / y])
Base.:/(x::Number, y::RevNode{T}) where {T} = new_node(:/, x / y.value, RevNode{T}[y], [-x / y.value^2])
Base.:^(x::RevNode{T}, n::Integer) where {T} = Base.power_by_squaring(x, n)
Base.tanh(x::RevNode{T}) where {T} = begin
    tanh_x = tanh(x.value)
    new_node(:tanh, tanh_x, RevNode{T}[x], [one(T) - tanh_x^2])
end
Base.exp(x::RevNode{T}) where {T} = begin
    exp_x = exp(x.value)
    new_node(:exp, exp_x, RevNode{T}[x], [exp_x])
end
Base.log(x::RevNode{T}) where {T} = new_node(:log, log(x.value), RevNode{T}[x], [one(T) / x.value])

function CGE.pullback!(f::RevNode{T}) where {T}
    for (parent, local_jac) in zip(f.args, f.metadata.local_jac)
        parent.metadata.derivative += f.metadata.derivative * local_jac
    end
end

"""Only supports scalar valued functions."""
function gradient!(f, g, x)
    inputs = map(RevNode{FloatType}, x)
    root = f(inputs)
    CGE.backward!(root)
    return map!(node -> node.metadata.derivative, g, inputs)
end

"""Only supports scalar valued functions."""
gradient(f, x) = gradient!(f, zero(x), x)

# --- jacobian and hessian --- #
export gradient, jacobian, hessian

"""Only supports vector valued functions."""
function jacobian(f, x)
    # J_ij = df_i/dx_j
    inputs = map(RevNode{FloatType}, x)
    outputs = f(inputs)
    jac = zeros(eltype(x), length(outputs), length(x))
    for (i, output) in enumerate(outputs)
        # Reset the derivatives for the next output
        for input in inputs
            input.metadata.derivative = zero(input.metadata.derivative)
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
        direction[j] = one(eltype(x))
        duals = map(Forward.Dual, x, direction)
        inputs = map(RevNode{eltype(duals)}, duals) # use Dual generic type
        output = f(inputs)
        CGE.backward!(output)
        hess[:, j] = map(node -> node.metadata.derivative.derivative, inputs)
    end
    return hess
end

end


