module Reverse

import ComputationGraphExplorer as CGE

mutable struct ReverseData
    derivative::Float64
    local_jac::Vector{Float64}
end

CGE.metadata(::Type{ReverseData}, ::Float64) = ReverseData(0.0, Float64[])
CGE.metadata_rows(data::ReverseData) = [
    "r" => data.derivative,
    "J" => data.local_jac,
]

const Node = CGE.Node{Float64,ReverseData}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

function new_node(op, value, args::Vector{Node}, local_jac)
    return Node(op, args, Float64(value), ReverseData(0.0, Float64.(local_jac)))
end

Base.zero(::Node) = Node(0.0)
Base.:*(x::Node, y::Node) = new_node(:*, x.value * y.value, Node[x, y], [y.value, x.value])
Base.:*(x::Node, y::Number) = new_node(:*, x.value * y, Node[x], [y])
Base.:*(x::Number, y::Node) = new_node(:*, x * y.value, Node[y], [x])
Base.:+(x::Node, y::Node) = new_node(:+, x.value + y.value, Node[x, y], [1.0, 1.0])
Base.:+(x::Node, y::Number) = new_node(:+, x.value + y, Node[x], [1.0])
Base.:+(x::Number, y::Node) = new_node(:+, x + y.value, Node[y], [1.0])
Base.:-(x::Node, y::Node) = new_node(:-, x.value - y.value, Node[x, y], [1.0, -1.0])
Base.:-(x::Node, y::Number) = new_node(:-, x.value - y, Node[x], [1.0])
Base.:-(x::Number, y::Node) = new_node(:-, x - y.value, Node[y], [-1.0])
Base.:-(x::Node) = new_node(:-, -x.value, Node[x], [-1.0])
Base.:/(x::Node, y::Node) = new_node(:/, x.value / y.value, Node[x, y], [1 / y.value, -x.value / y.value^2])
Base.:/(x::Node, y::Number) = new_node(:/, x.value / y, Node[x], [1 / y])
Base.:/(x::Number, y::Node) = new_node(:/, x / y.value, Node[y], [-x / y.value^2])
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = new_node(:tanh, tanh(x.value), Node[x], [1 - tanh(x.value)^2])
Base.exp(x::Node) = new_node(:exp, exp(x.value), Node[x], [exp(x.value)])
Base.log(x::Node) = new_node(:log, log(x.value), Node[x], [1 / x.value])

function CGE.pullback!(f::Node)
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

#"""Only supports scalar valued functions."""
#function hessian(f, x)
#    inputs = map(Forward.Dual, x)
#    return Base.Forward.jacobian(z -> gradient(f, z, x_dual))
#end

end


