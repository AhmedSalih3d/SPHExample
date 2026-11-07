module BoatGeometryTests
using Test, SPHExampleGPU, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__,"..","example","Boat3d.jl"))

function box_faces(lo,hi)
    v = [SVector(x,y,z) for x in (lo[1],hi[1]) for y in (lo[2],hi[2]) for z in (lo[3],hi[3])]
    # Each quad points outward; two triangles share one diagonal.
    quads = ((1,2,4,3),(5,7,8,6),(1,5,6,2),(3,4,8,7),(1,3,7,5),(2,6,8,4))
    return [(v[q[1]],v[q[j]],v[q[j+1]]) for q in quads for j in 2:3]
end
function write_ascii(path,faces)
    open(path,"w") do io
        println(io,"solid test")
        for face in faces
            println(io,"facet normal 0 0 0\nouter loop")
            for p in face
                println(io,"vertex ",join(p," "))
            end
            println(io,"endloop\nendfacet")
        end
        println(io,"endsolid test")
    end
end

@testset "STL volume, geometry and packing" begin
    faces = box_faces((0,0,0),(1,1,1))
    mesh = TriangleSurface(faces)
    @test signed_distance(mesh,(0.5,0.5,0.5)) ≈ -0.5
    @test signed_distance(mesh,(2,2,2)) ≈ sqrt(3)
    @test signed_distance(mesh,(0.5,0.5,0)) == 0
    @test_throws ArgumentError TriangleSurface(faces[2:end])
    @test_throws ArgumentError TriangleSurface([faces;[faces[1]]])
    @test_throws ArgumentError TriangleSurface([(faces[1][1],faces[1][1],faces[1][2])])
    # A rotated surface exercises the closest-face and BVH paths away from axes.
    rotation = @SMatrix [cos(0.37) -sin(0.37) 0; sin(0.37) cos(0.37) 0; 0 0 1]
    rotated = TriangleSurface([Tuple(rotation*p for p in f) for f in faces])
    for p in (SVector(0.5,0.5,0.5),SVector(1.3,0.2,0.4),SVector(-0.2,-0.3,0.2))
        @test signed_distance(rotated,rotation*p) ≈ signed_distance(mesh,p) atol=1e-12
    end
    packed = sample_solid(rotated,0.1;iterations=10)
    @test packed.quality.minimum_clearance >= 0.05-1e-9
    @test packed.quality.minimum_spacing > 0.065
    @test 0.8 < packed.quality.represented_volume < 1.2
    @test_throws ArgumentError sample_solid(mesh,0)
    @test_throws ArgumentError sample_solid(mesh,0.1;layers=0)
    mktempdir() do dir
        ascii = joinpath(dir,"box.stl")
        write_ascii(ascii,faces)
        @test signed_distance(load_stl(ascii;scale=2,translation=(1,0,0)),(2,1,1)) ≈ -1
        binary = joinpath(dir,"binary.stl")
        open(binary,"w") do io
            write(io,zeros(UInt8,80)); write(io,htol(UInt32(length(faces))))
            for f in faces
                write(io,zeros(UInt8,12))
                for p in f, x in p
                    write(io,htol(reinterpret(UInt32,Float32(x))))
                end
                write(io,UInt16(0))
            end
        end
        @test load_stl(binary).faces == mesh.faces
        hull_file = joinpath(dir,"hull.stl")
        write_ascii(hull_file,box_faces((0,-0.1,0),(0.4,0.1,0.2)))
        setup = boat_setup(hull_file;dx=0.02,duration=0.001,packing_iterations=2,save_location=dir)
        @test setup.constants.m₀ ≈ 1000*0.02^3
        @test setup.packing.minimum_clearance >= 0.01-1e-9
        @test all(g -> g.Type != Fluid || all(p -> signed_distance(setup.mesh,p) >= 0.01-1e-9,g.Particles.Position),setup.geometry)
        @test setup.metadata.Periodicity === nothing
    end
end
end
