module MDBCGhostDataConfiguration

export MDBCGhostData, ResetGhostData!, InitializeGhostData!, EnsureGhostNeighborCellListsSize!

"""
    MDBCGhostData

Stores active ghost-particle indices and their neighboring cell lists used by MDBC.
"""
struct MDBCGhostData
    Indices::Vector{Int}
    NeighborCellLists::Vector{Vector{Int}}
end

MDBCGhostData() = MDBCGhostData(Int[], Vector{Vector{Int}}())

function ResetGhostData!(GhostData::MDBCGhostData)
    empty!(GhostData.Indices)
    empty!(GhostData.NeighborCellLists)
    return nothing
end

function InitializeGhostData!(GhostData::MDBCGhostData, GhostIndices::AbstractVector{Int})
    indices = GhostData.Indices
    empty!(indices)
    sizehint!(indices, length(GhostIndices))
    append!(indices, GhostIndices)
    EnsureGhostNeighborCellListsSize!(GhostData)
    return nothing
end

function EnsureGhostNeighborCellListsSize!(GhostData::MDBCGhostData)
    target_len = length(GhostData.Indices)
    neighbor_cell_lists = GhostData.NeighborCellLists
    original_len = length(neighbor_cell_lists)
    resize!(neighbor_cell_lists, target_len)

    if target_len > original_len
        @inbounds for idx in (original_len + 1):target_len
            neighbor_cell_lists[idx] = Int[]
        end
    end

    return nothing
end

end
