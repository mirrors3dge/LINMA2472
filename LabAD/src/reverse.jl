module Reverse

import ComputationGraphExplorer as CGE

mutable struct ReverseData
    derivative::Float64
    parents::Union{Nothing,Vector{Tuple{Any,Float64}}} # Any to avoid circular def with Node
end
ReverseData(derivative) = ReverseData(derivative, nothing)

CGE.metadata(::Type{ReverseData}, ::Float64) = ReverseData(0.0)
CGE.metadata_rows(data::ReverseData) = ["r" => data.derivative]

const Node = CGE.Node{Float64,ReverseData}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

# ops impls
local_jacobian(::typeof(+), args::Node...) = [(a, 1.0) for a in args]
local_jacobian(::typeof(-), x::Node, y::Node) = [(x, 1.0), (y, -1.0)]
local_jacobian(::typeof(-), x::Node) = [(x, -1.0)]
local_jacobian(::typeof(*), x::Node, y::Node) = [(x, y.value), (y, x.value)]
local_jacobian(::typeof(/), x::Node, y::Node) = [(x, 1 / y.value), (y, -x.value / y.value^2)]
local_jacobian(::typeof(^), x::Node, n::Node) = [(x, n.value * x.value^(n.value - 1))]
local_jacobian(::typeof(tanh), x::Node) = [(x, 1 - tanh(x.value)^2)]
local_jacobian(::typeof(exp), x::Node) = [(x, exp(x.value))]
local_jacobian(::typeof(log), x::Node) = [(x, 1 / x.value)]

function CGE.pullback!(op, f::Node, args::Node...)
    if f.metadata.parents === nothing
        f.metadata.parents = local_jacobian(op, args...)
    end
    for (parent, coef) in f.metadata.parents
        parent.metadata.derivative += f.metadata.derivative * coef
    end
end

function gradient!(f, g, x)
    x_nodes = map(Node, x)
    expr = f(x_nodes)
    CGE.backward!(expr)
    return map!(node -> node.metadata.derivative, g, x_nodes)
end

gradient(f, x) = gradient!(f, zero(x), x)

end
