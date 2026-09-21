module Forward

struct Dual
    value::Float64
    derivative::Float64
end
Dual(x::Number, y::Number) = Dual(Float64(x), Float64(y))

Base.broadcastable(d::Dual) = Ref(d)
Base.zero(::Dual) = Dual(0, 0)
Base.zero(::Type{Dual}) = Dual(0, 0)
# Addition and subtraction
Base.:+(x::Dual, y::Dual) = Dual(x.value + y.value, x.derivative + y.derivative)
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
# Specific functions and operations
Base.tanh(x::Dual) = Dual(tanh(x.value), (1 - tanh(x.value)^2) * x.derivative)
Base.exp(x::Dual) = Dual(exp(x.value), x.derivative * exp(x.value))
Base.log(x::Dual) = Dual(log(x.value), x.derivative / x.value)

Base.show(io::IO, d::Dual) = print(io, "Dual(", d.value, ", ", d.derivative, ")")

Base.max(x::Dual, y::Real) = if (x.value > y) x.value else y end
Base.max(x::Real, y::Dual) = if (x > y.value) x else y.value end

Base.isless(x::Dual, y::Real) = x.value < y
Base.isless(x::Real, y::Dual) = x < y.value
Base.isless(x::Dual, y::Dual) = x.value < y.value

# fixme
#Base.relu(x::Dual) = Dual(if (x.value > 0) x.value else 0 end, if (x.value > 0) 1 else 0 end)

function onehot(v, i)
    z = zero(similar(v, Float64))
    z[i] = 1.0
    return z
end

function gradient(f, x, i::Integer)
    dx = map(Dual, x, onehot(x, i))
    return f(dx).derivative
end

function gradient!(f, g, x)
    return map!(g, eachindex(x)) do i
        gradient(f, x, i)
    end
end

gradient(f, x) = gradient!(f, zero(x), x)

end
