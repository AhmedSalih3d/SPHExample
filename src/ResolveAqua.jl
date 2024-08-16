__precompile__(false)

using SentinelArrays
using StructArrays: StructArray
using InlineStrings

# Resolving ambiguity #1
Base.:<(a::SentinelArrays.ChainedVectorIndex, b::BigInt) = <(Int(a), b)

# Resolving ambiguity #2
Base.:<(a::BigInt, b::SentinelArrays.ChainedVectorIndex) = <(a, Int(b))

# Resolving ambiguity #3
Base.:<=(a::BigInt, b::SentinelArrays.ChainedVectorIndex) = <=(a, Int(b))

# Resolving ambiguity #4
Base.:<=(a::SentinelArrays.ChainedVectorIndex, b::BigInt) = <=(Int(a), b)

# Resolving ambiguity #5
Base.:(==)(a::SentinelArrays.ChainedVectorIndex, b::BigInt) = ==(BigInt(a), b)

# Resolving ambiguity #6
Base.:(==)(a::BigInt, b::SentinelArrays.ChainedVectorIndex) = ==(a, BigInt(b))

# Resolving ambiguity #7
Base.Broadcast.broadcasted(f::F, A::SentinelArrays.ChainedVector) where F = Base.Broadcast.broadcasted(f, collect(A))

# Resolving ambiguity #8
Base.copyto!(dest::PermutedDimsArray{T, 1}, src::SentinelArrays.ChainedVector) where T = Base.copyto!(dest, collect(src))

# Resolving ambiguity #9
Base.copyto!(dest::PermutedDimsArray{T, 1}, src::SentinelArrays.ChainedVector{T, A} where A<:AbstractVector{T}) where T = Base.copyto!(dest, collect(src))

# Resolving ambiguity #11
Base.findall(f::Function, x::SentinelArrays.ChainedVector) = Base.findall(f, collect(x))

# Resolving ambiguity #12
Base.reduce(::typeof(vcat), x::SentinelArrays.ChainedVector{T, A} where {T<:(AbstractVecOrMat), A<:AbstractVector{T}}) = Base.reduce(vcat, collect(x))

# Resolving ambiguity #13
Base.reduce(::typeof(hcat), x::SentinelArrays.ChainedVector{T, A} where {T<:(AbstractVecOrMat), A<:AbstractVector{T}}) = Base.reduce(hcat, collect(x))

# Resolving ambiguity #14
Base.reshape(s::StructArray{T}, d::Tuple{Integer, Vararg{Integer}}) where T = Base.reshape(collect(s), d)

# Resolving ambiguity #15
Base.similar(s::StructArray, S::Type, sz::Tuple{Integer, Vararg{Integer}}) = Base.similar(collect(s), S, sz)

# Resolving ambiguity #16
Base.similar(s::StructArray, S::Type, sz::Tuple{Integer, Vararg{Integer}}) = Base.similar(collect(s), S, sz)

# Resolving Ambiguity #1
Base.Broadcast.broadcasted(::Base.Broadcast.BroadcastStyle, f::F, A::SentinelArrays.ChainedVector) where {F} = Base.Broadcast.broadcasted(f, collect(A))

# Resolving Ambiguity #2
Base.Broadcast.broadcasted(::Type{InlineStrings.InlineString}, A::SentinelArrays.ChainedVector) = Base.Broadcast.broadcasted(InlineStrings.InlineString, collect(A))

# Resolving Ambiguity #3
Base.Sort.defalg(::AbstractArray{<:Missing}) = Base.Sort.defalg(collect(Missing[]))

# Correct way to resolve the ambiguity
Base.Broadcast.broadcasted(::F1, ::F2) where {F1<:Base.Broadcast.BroadcastStyle, F2<:SentinelArrays.ChainedVector} = Base.Broadcast.broadcasted(F1, collect(F2))

# Resolving Ambiguity #4
Base.findall(pred::Base.Fix2{typeof(in)}, x::SentinelArrays.ChainedVector) = Base.findall(pred, collect(x))
