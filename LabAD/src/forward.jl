module Forward

const FloatType = Float64

struct Dual{T}
    value::T
    derivative::T
end
Dual(x::Number, y::Number) = Dual{FloatType}(FloatType(x), FloatType(y))
Base.convert(::Type{Dual{T}}, x::Number) where {T} = Dual{T}(T(x), zero(T))

Base.broadcastable(d::Dual) = Ref(d)
Base.zero(::Dual{T}) where {T} = Dual{T}(zero(T), zero(T))
Base.zero(::Type{Dual{T}}) where {T} = Dual{T}(zero(T), zero(T))
Base.one(::Dual{T}) where {T} = Dual{T}(one(T), zero(T))
Base.one(::Type{Dual{T}}) where {T} = Dual{T}(one(T), zero(T))
# Addition and subtraction
Base.:+(x::Dual, y::Dual) = Dual(x.value + y.value, x.derivative + y.derivative)
Base.:+(x::Dual, y::Number) = Dual(x.value + y, x.derivative)
Base.:+(x::Number, y::Dual) = Dual(x + y.value, y.derivative)
Base.:-(x::Dual, y::Dual) = Dual(x.value - y.value, x.derivative - y.derivative)
Base.:-(x::Dual, y::Number) = Dual(x.value - y, x.derivative)
Base.:-(x::Number, y::Dual) = Dual(x - y.value, -y.derivative)
Base.:-(x::Dual) = Dual(-x.value, -x.derivative)
# Scalar multiplication and division
Base.:*(α::Number, x::Dual) = Dual(α * x.value, α * x.derivative)
Base.:*(x::Dual, α::Number) = Dual(x.value * α, x.derivative * α)
Base.:/(x::Dual, α::Number) = Dual(x.value / α, x.derivative / α)
# Dual multiplication, division and power
Base.:*(x::Dual, y::Dual) = Dual(x.value * y.value, x.value * y.derivative + x.derivative * y.value)
Base.:/(x::Dual, y::Dual) = Dual(x.value / y.value, (x.derivative * y.value - x.value * y.derivative) / y.value^2)
Base.:/(α::Number, x::Dual) = Dual(α / x.value, -α * x.derivative / x.value^2)
Base.:^(x::Dual, n::Integer) = Base.power_by_squaring(x, n)
# Specific functions and operations`
Base.tanh(x::Dual) = Dual(tanh(x.value), (1 - tanh(x.value)^2) * x.derivative)
Base.exp(x::Dual) = Dual(exp(x.value), exp(x.value) * x.derivative)
Base.log(x::Dual) = Dual(log(x.value), x.derivative / x.value)
# relu
Base.isless(x::Dual, y::Number) = x.value < y
Base.isless(x::Number, y::Dual) = x < y.value
Base.isless(x::Dual, y::Dual) = x.value < y.value
Base.max(x::Dual, y::Number) =
    if (x.value > y) x else Dual(y, 0) end
Base.max(x::Number, y::Dual) =
    if (x > y.value) Dual(x, 0) else y end
Base.max(x::Dual, y::Dual) =
    if (x > y) x else y end

# fixme (not needed?)
#function relu(x::Dual)
#    x.value > 0 ? x : Dual(0, 0)
#end

# display
Base.show(io::IO, d::Dual) = print(io, "Dual(", d.value, ", ", d.derivative, ")")

function onehot(v, i)
    z = zero(v)
    z[i] = one(z[i])
    return z
end

"""Only supports scalar valued functions."""
function gradient(f, x, i::Integer)
    inputs = map(Dual, x, onehot(x, i))
    return f(inputs).derivative
end

"""Only supports scalar valued functions."""
function gradient!(f, g, x)
    return map!(g, eachindex(x)) do i
        gradient(f, x, i)
    end
end

"""Only supports scalar valued functions."""
gradient(f, x) = gradient!(f, zero(x), x)

"""
Jacobian-vector product `J(x) * direction`.

Only supports vector valued functions.
"""
function forward_deriv(f, x, direction)
    inputs = map(Dual, x, direction)
    return map(d -> d.derivative, f(inputs))
end

# --- jacobian and hessian --- #
export gradient, forward_deriv, jacobian, hessian, hvp

"""Only supports vector valued functions."""
function jacobian(f, x)
    # J_ij = df_i/dx_j
    J_ij = map(j -> forward_deriv(f, x, onehot(x, j)), eachindex(x))
    return hcat(J_ij...) # format as matrix
end

"""
Using forward-over-forward (slooow).

Only supports scalar valued functions.
"""
function hessian(f, x)
    return jacobian(z -> gradient(f, z), x)
end

"""
Hessian-vector product using forward-over-forward (slooow).

Only supports scalar valued functions.
"""
function hvp(f, x, direction)
    return forward_deriv(z -> gradient(f, z), x, direction)
end

end # module Forward
