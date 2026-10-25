#
# vehicle_3d_geometry.jl
#
# Mesh primitives and vehicle body construction shared by the 3D visualisers
# (visualize_vehicle_3d.jl, visualize_dlc_3d.jl). Pure geometry: no simulation,
# no plotting backend, no module-level state.
#

using LinearAlgebra

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

"Rotation about the world z axis."
rotz(psi) = [cos(psi) -sin(psi) 0.0; sin(psi) cos(psi) 0.0; 0.0 0.0 1.0]

"""
Body-to-world rotation for yaw `psi`, pitch `theta` (positive drops the nose),
roll `phi` (positive lifts the left side).
"""
function rotypr(psi, theta, phi)
    cps, sps = cos(psi), sin(psi)
    cth, sth = cos(theta), sin(theta)
    cph, sph = cos(phi), sin(phi)
    return [cps*cth  cps*sth*sph-sps*cph  cps*sth*cph+sps*sph;
            sps*cth  sps*sth*sph+cps*cph  sps*sth*cph-cps*sph;
               -sth            cth*sph              cth*cph   ]
end

# ---------------------------------------------------------------------------
# Meshes. Each returns (x, y, z, i, j, k) with 0-based triangle indices, which
# is what Plotly's mesh3d expects.
# ---------------------------------------------------------------------------

"Box of half-extents `half`, placed at `centre` with orientation `R`."
function box_mesh(centre, R, half)
    signs = ((-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
             (-1,-1, 1), (1,-1, 1), (1,1, 1), (-1,1, 1))
    pts = [centre .+ R * [s[1]*half[1], s[2]*half[2], s[3]*half[3]] for s in signs]
    return ([p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts],
            [0,0, 4,4, 0,0, 1,1, 2,2, 3,3],
            [1,2, 5,6, 1,5, 2,6, 3,7, 0,4],
            [2,3, 6,7, 5,4, 6,5, 7,6, 4,7])
end

"Cylinder of radius `radius` and width `width`, spin axis along the local y axis."
function cylinder_mesh(centre, R, radius, width; n = 20)
    ths = range(0, 2pi; length = n + 1)[1:n]
    pts = Vector{Vector{Float64}}()
    for s in (-1.0, 1.0), t in ths
        push!(pts, centre .+ R * [radius*cos(t), s*width/2, radius*sin(t)])
    end
    for s in (-1.0, 1.0)                      # cap centres
        push!(pts, centre .+ R * [0.0, s*width/2, 0.0])
    end
    I, J, K = Int[], Int[], Int[]
    for q in 0:n-1
        q2 = (q + 1) % n
        push!(I, q);      push!(J, q2);     push!(K, n + q)
        push!(I, q2);     push!(J, n + q2); push!(K, n + q)
        push!(I, 2n);     push!(J, q);      push!(K, q2)
        push!(I, 2n + 1); push!(J, n + q);  push!(K, n + q2)
    end
    return ([p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts], I, J, K)
end

"Upright cone of base `radius` and height `h`, apex up, base centred at `base`."
function cone_mesh(base, radius, h; n = 12)
    ths = range(0, 2pi; length = n + 1)[1:n]
    pts = [[base[1] + radius*cos(t), base[2] + radius*sin(t), base[3]] for t in ths]
    push!(pts, [base[1], base[2], base[3] + h])    # apex
    push!(pts, [base[1], base[2], base[3]])        # base centre
    I, J, K = Int[], Int[], Int[]
    for q in 0:n-1
        q2 = (q + 1) % n
        push!(I, q); push!(J, q2); push!(K, n)       # side
        push!(I, q); push!(J, q2); push!(K, n + 1)   # base
    end
    return ([p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts], I, J, K)
end

"""
Torus swept around the circle of radius `R_rim` lying in the plane spanned by
`v` (wheel-frame right) and `u` (wheel-frame up), with tube radius `r_tube`.
`theta` rotates the ring in its own plane, measured from `u` towards `-v`.
"""
function torus_mesh(centre, v, u, n, R_rim, r_tube, theta; nmaj = 44, nmin = 9)
    X, Y, Z = Float64[], Float64[], Float64[]
    for a in 0:nmaj-1
        phi = 2pi*a/nmaj
        rad = sin(phi - theta) .* v .+ cos(phi - theta) .* u
        rim = centre .+ R_rim .* rad
        for b in 0:nmin-1
            psi = 2pi*b/nmin
            p = rim .+ r_tube .* (cos(psi) .* rad .+ sin(psi) .* n)
            push!(X, p[1]); push!(Y, p[2]); push!(Z, p[3])
        end
    end
    I, J, K = Int[], Int[], Int[]
    for a in 0:nmaj-1, b in 0:nmin-1
        a2 = (a+1) % nmaj; b2 = (b+1) % nmin
        i00 = a*nmin + b; i10 = a2*nmin + b; i01 = a*nmin + b2; i11 = a2*nmin + b2
        push!(I, i00); push!(J, i10); push!(K, i11)
        push!(I, i00); push!(J, i11); push!(K, i01)
    end
    return (X, Y, Z, I, J, K)
end

"""
    steering_wheel_frame(tilt)

Orthonormal (right, up, column-axis) triad for a steering wheel on a car facing
world +X, with the column raked back by `tilt` from horizontal. `v` points to
the driver's right (world -Y), so a positive rotation from `u` towards `-v`
carries the top of the rim to the driver's left — matching the model's sign
convention, in which positive road-wheel `delta` steers left.
"""
function steering_wheel_frame(tilt)
    n = [-cos(tilt), 0.0, sin(tilt)]           # column axis, back and up
    u = normalize([0.0, 0.0, 1.0] - dot([0.0, 0.0, 1.0], n) .* n)
    v = cross(u, n)
    return v, u, n
end

"""
    steering_wheel_meshes(centre, sw_angle; ...)

Rim, hub, three spokes and a 12-o'clock rim marker, as `(mesh, colour)` pairs.
`sw_angle` is the steering-wheel angle in radians, positive for a left turn.
"""
function steering_wheel_meshes(centre, sw_angle; radius = 0.95, tilt = deg2rad(25),
                               tube = 0.075, spoke_w = 0.075, spoke_t = 0.05,
                               rim_color = "rgb(35,35,40)", hub_color = "rgb(90,90,98)",
                               mark_color = "rgb(220,45,45)",
                               nmaj = 30, nmin = 7)
    v, u, n = steering_wheel_frame(tilt)
    parts = Tuple{Any,String}[]
    # The rim dominates the size of an animated scene: every vertex is re-sent
    # on every frame, so keep the tessellation only as fine as it needs to look
    # round at this radius.
    push!(parts, (torus_mesh(centre, v, u, n, radius, tube, sw_angle; nmaj, nmin), rim_color))
    push!(parts, (cylinder_mesh(centre, hcat(u, n, v), radius*0.17, tube*1.3), hub_color))
    for phi in (pi/2, pi, 3pi/2)               # 3, 6 and 9 o'clock spokes
        d = sin(phi - sw_angle) .* v .+ cos(phi - sw_angle) .* u
        push!(parts, (box_mesh(centre .+ d .* (radius/2), hcat(cross(d, n), d, n),
                               (spoke_w/2, radius/2, spoke_t/2)), hub_color))
    end
    d0 = sin(-sw_angle) .* v .+ cos(-sw_angle) .* u
    push!(parts, (box_mesh(centre .+ d0 .* radius, hcat(cross(d0, n), d0, n),
                           (tube*1.5, tube*1.7, tube*1.5)), mark_color))
    return parts
end

"Fixed 12-o'clock reference tick just outside the rim; it does not rotate."
function steering_tick_mesh(centre; radius = 0.95, tilt = deg2rad(25), tube = 0.075)
    v, u, n = steering_wheel_frame(tilt)
    return box_mesh(centre .+ u .* (radius + tube*2.4), hcat(v, u, n),
                    (tube*0.45, tube*1.3, tube*0.9))
end

"Blue (light) to red (heavy) ramp, used for wheel vertical load."
function load_color(Fz, lo, hi)
    u = clamp((Fz - lo) / max(hi - lo, eps()), 0.0, 1.0)
    r = round(Int, 255 * (0.20 + 0.75u))
    g = round(Int, 255 * (0.55 - 0.42u))
    b = round(Int, 255 * (0.88 - 0.74u))
    return "rgb($r,$g,$b)"
end

# ---------------------------------------------------------------------------
# Vehicle body
# ---------------------------------------------------------------------------

"Visual proportions of the modelled car. Axle positions and track match the model."
Base.@kwdef struct VehicleGeom
    a::Float64          = 1.2      # CG to front axle
    b::Float64          = 1.4      # CG to rear axle
    track::Float64      = 1.55
    wheel_r::Float64    = 0.32
    wheel_w::Float64    = 0.22
    body_l::Float64     = 4.40
    body_w::Float64     = 1.76
    body_h::Float64     = 0.62
    body_z::Float64     = 0.78     # body-box centre height
    cabin_l::Float64    = 1.90
    cabin_w::Float64    = 1.52
    cabin_h::Float64    = 0.52
    cabin_x::Float64    = -0.25
    front_over::Float64 = 0.90     # front overhang beyond the front axle
end

"Wheel station positions (x ahead of CG, y left of CG): FL, FR, RL, RR."
stations(g::VehicleGeom) = ((g.a,  g.track/2), (g.a, -g.track/2),
                            (-g.b, g.track/2), (-g.b, -g.track/2))

"World-frame (x,y) of the four body corners, for corridor scoring and outlines."
function body_corners(g::VehicleGeom, X, Y, psi)
    c, s = cos(psi), sin(psi)
    x_off = (g.a + g.front_over) - g.body_l/2
    out = NTuple{2,Float64}[]
    for sx in (-1, 1), sy in (-1, 1)
        bx = x_off + sx*g.body_l/2
        by = sy*g.body_w/2
        push!(out, (X + c*bx - s*by, Y + s*bx + c*by))
    end
    return out
end

"""
    car_meshes(geom, X, Y, psi, roll, pitch, delta, Fz, Fz_lo, Fz_hi)

Meshes for body shell, cabin and four wheels, plus the wheel colours encoding
vertical load. The sprung mass yaws, pitches and rolls; the wheels only yaw and
steer, because the unsprung mass stays flat on the road.
"""
function car_meshes(g::VehicleGeom, X, Y, psi, roll, pitch, delta, Fz, Fz_lo, Fz_hi)
    pos = [X, Y, 0.0]
    Rb  = rotypr(psi, pitch, roll)
    Ry  = rotz(psi)

    body_centre  = pos + Rb * [(g.a + g.front_over) - g.body_l/2, 0.0, g.body_z]
    cabin_centre = pos + Rb * [g.cabin_x, 0.0, g.body_z + g.body_h/2 + g.cabin_h/2]

    meshes = Any[box_mesh(body_centre,  Rb, (g.body_l/2,  g.body_w/2,  g.body_h/2)),
                 box_mesh(cabin_centre, Rb, (g.cabin_l/2, g.cabin_w/2, g.cabin_h/2))]
    colors = String[]
    for (q, (xs, ys)) in enumerate(stations(g))
        steer  = q <= 2 ? delta : 0.0
        centre = pos + Ry * [xs, ys, g.wheel_r]
        push!(meshes, cylinder_mesh(centre, Ry * rotz(steer), g.wheel_r, g.wheel_w))
        push!(colors, load_color(Fz[q], Fz_lo, Fz_hi))
    end
    return meshes, colors
end
