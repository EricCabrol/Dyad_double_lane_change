#
# visualize_dlc_3d.jl
#
# Interactive 3D animation of the closed-loop double lane change: the MPC driver
# taking the double-track vehicle through the cone gates.
#
# Run from the package root:
#     julia --project=. scripts/visualize_dlc_3d.jl
#
# Produces, in this directory:
#   dlc_3d_follow.html     animated, camera tracking the car, at the max pass speed
#   dlc_3d_overview.html   whole course at once, ghost cars along the path
#   dlc_3d_fail.html       animated run just above the limit, struck cones in red
#
# Rendering uses PlotlyBase, which ships with the Dyad runtime.
#
# ---------------------------------------------------------------------------
# What is simulated and what is drawn
#
# Simulated: CG trajectory, heading, steer angle, per-corner vertical load
# (wheel colour), speed. The cones are the real scoring geometry — a cone turns
# red when a body corner passes within its radius, which is the same corner
# test `score` uses in dlc_mpc.jl.
#
# Drawn only: the body roll and pitch. The model has three degrees of freedom
# and no roll or pitch state; the attitude shown is reconstructed from ax_lag
# and ay_lag, the suspension-lagged accelerations that drive load transfer in
# the model. Set ROLL_DEG_PER_G / PITCH_DEG_PER_G to zero to remove it.
# ---------------------------------------------------------------------------

include(joinpath(@__DIR__, "dlc_mpc.jl"))
include(joinpath(@__DIR__, "vehicle_3d_geometry.jl"))

using PlotlyBase
const PB = PlotlyBase

const GEOM            = VehicleGeom()
const ROLL_DEG_PER_G  = 4.5
const PITCH_DEG_PER_G = 1.8
const WINDOW          = 18.0          # follow-camera half-window [m]
const Z_RANGE         = (0.0, 3.6)
const CONE_SPACING    = 3.0           # along each gate edge [m]
const CONE_R          = 0.17
const CONE_H          = 0.55

const BODY_COLOR  = "rgb(38,80,140)"
const CABIN_COLOR = "rgb(150,185,225)"
const CONE_OK     = "rgb(240,130,30)"
const CONE_HIT    = "rgb(210,25,25)"

# --- steering-wheel gauge --------------------------------------------------
# STEER_RATIO comes from dlc_mpc.jl, so the gauge, the live readout and the
# static figure can never disagree about what a steering-wheel angle means.
# Gauge position, relative to the CG in WORLD axes. Its orientation stays fixed
# while the car yaws, so the rim rotation you see is the steering angle alone
# and not steering plus yaw.
const SW_OFFSET = [-1.5, -7.0, 2.1]
const SW_RADIUS = 0.95

# --- 2D time-history panels ------------------------------------------------
# The 3D scene is given a domain so the right-hand strip is free for 2D axes.
const SCENE_RIGHT = 0.66
const PANEL_X     = [0.745, 0.985]
const PANEL_Y_SW  = [0.575, 0.945]
const PANEL_Y_PSI = [0.085, 0.455]

const OUT_FOLLOW   = joinpath(@__DIR__, "dlc_3d_follow.html")
const OUT_OVERVIEW = joinpath(@__DIR__, "dlc_3d_overview.html")
const OUT_FAIL     = joinpath(@__DIR__, "dlc_3d_fail.html")

# ===========================================================================
# Cones
# ===========================================================================

"""
Cone base positions along both edges of every gate.

The cone *centre* sits `CONE_R` outside the corridor boundary, so the cone's
inner face is tangent to it. That is both how cones are really laid out and
what makes the red-cone indicator agree with `score`: for a corner at lateral
margin `m` from the boundary, the distance to the cone centre is `CONE_R + m`,
so it falls inside `CONE_R` exactly when `m < 0`.
"""
function cone_positions(course::DLCCourse)
    pts = NTuple{2,Float64}[]
    for g in gates(course)
        n = max(2, round(Int, (g.x1 - g.x0)/CONE_SPACING) + 1)
        for x in range(g.x0, g.x1; length = n), sy in (-1, 1)
            push!(pts, (x, g.yc + sy*(g.half + CONE_R)))
        end
    end
    return pts
end

"""
    struck(cones, traj)

Which cones a body corner passes within `CONE_R` of, using the same corner
geometry as `score`.

Laterally this matches the margin test exactly. Longitudinally it does not:
cones are discrete at `CONE_SPACING`, so a shallow incursion that happens
between two cones is a negative margin with no cone struck. Measured on this
course, the run at +0.5 m/s over the limit has a margin of -0.06 m and hits
nothing. The margin plot in dlc_mpc.jl is the continuous test; the cones are
the physical one, and the physical one is more forgiving — which is exactly
why real moose tests are scored on cones and still have a tolerance debate.
"""
function struck(cones, traj)
    hit = falses(length(cones))
    for n in eachindex(traj.t)
        cs = body_corners(GEOM, traj.X[n], traj.Y[n], traj.psi[n])
        for (ci, (cx, cy)) in enumerate(cones)
            hit[ci] && continue
            for (bx, by) in cs
                if hypot(bx - cx, by - cy) < CONE_R
                    hit[ci] = true; break
                end
            end
        end
    end
    return hit
end

# ===========================================================================
# Motion sampling
# ===========================================================================

"Run the closed loop and resample everything the scene needs onto `nframes`."
function sample_dlc(; vx0, course = DLCCourse(), path, mpc = MPCParams(), nframes = 170)
    traj = run_dlc(build_plant(:doubletrack), vx0; course, path, mpc, record_corners = true)
    isempty(traj.t) && error("run produced no samples")
    g = 9.81
    idx = round.(Int, range(1, length(traj.t); length = min(nframes, length(traj.t))))
    pick(v) = v[idx]
    return (; t = pick(traj.t), X = pick(traj.X), Y = pick(traj.Y), psi = pick(traj.psi),
             vx = pick(traj.vx), vy = pick(traj.vy), r = pick(traj.r),
             ay = pick(traj.ay), beta = pick(traj.beta), delta = pick(traj.delta),
             roll  =  deg2rad(ROLL_DEG_PER_G)  .* pick(traj.ay_lag) ./ g,
             pitch = -deg2rad(PITCH_DEG_PER_G) .* pick(traj.ax_lag) ./ g,
             Fz = [pick(traj.Fz[q]) for q in 1:4],
             full = traj)
end

# ===========================================================================
# Scene
# ===========================================================================

"Static traces: road, grid, cones, reference, driven path, tyre tracks."
function static_traces(M, course, path, cones, hit)
    tr = PB.AbstractTrace[]
    xlo, xhi = -15.0, total_length(course) + 25.0
    ylo, yhi = -12.0, course.D + 12.0

    push!(tr, PB.mesh3d(x = [xlo,xhi,xhi,xlo], y = [ylo,ylo,yhi,yhi], z = fill(-0.02, 4),
                        i = [0,0], j = [1,2], k = [2,3], color = "rgb(238,238,236)",
                        flatshading = true, hoverinfo = "skip", showscale = false,
                        showlegend = false, name = "road"))

    gx, gy, gz = Float64[], Float64[], Float64[]
    for xg in ceil(xlo/10)*10 : 10.0 : xhi
        append!(gx, [xg, xg, NaN]); append!(gy, [ylo, yhi, NaN]); append!(gz, [0.0,0.0,NaN])
    end
    for yg in ceil(ylo/10)*10 : 10.0 : yhi
        append!(gx, [xlo, xhi, NaN]); append!(gy, [yg, yg, NaN]); append!(gz, [0.0,0.0,NaN])
    end
    push!(tr, PB.scatter3d(x = gx, y = gy, z = gz, mode = "lines",
                           line = PB.attr(color = "rgb(210,210,208)", width = 1),
                           hoverinfo = "skip", showlegend = false, name = "10 m grid"))

    # cones, batched into one mesh per colour so the scene stays light
    for (lbl, col, want) in (("cones", CONE_OK, false), ("cones struck", CONE_HIT, true))
        X, Y, Z, I, J, K = Float64[], Float64[], Float64[], Int[], Int[], Int[]
        for (ci, (cx, cy)) in enumerate(cones)
            hit[ci] == want || continue
            x, y, z, i, j, k = cone_mesh([cx, cy, 0.0], CONE_R, CONE_H)
            off = length(X)
            append!(X, x); append!(Y, y); append!(Z, z)
            append!(I, i .+ off); append!(J, j .+ off); append!(K, k .+ off)
        end
        isempty(X) && continue
        push!(tr, PB.mesh3d(x = X, y = Y, z = Z, i = I, j = J, k = K, color = col,
                            flatshading = true, hoverinfo = "skip", showscale = false,
                            showlegend = true, name = lbl))
    end

    ss = 0:0.5:total_length(course)
    push!(tr, PB.scatter3d(x = collect(ss), y = [y_ref(path, s) for s in ss],
                           z = fill(0.03, length(ss)), mode = "lines",
                           line = PB.attr(color = "rgb(230,90,40)", width = 4, dash = "dash"),
                           hoverinfo = "skip", name = "MPC reference"))

    push!(tr, PB.scatter3d(x = M.full.X, y = M.full.Y, z = fill(0.06, length(M.full.X)),
                           mode = "lines",
                           line = PB.attr(color = M.full.ay, colorscale = "Viridis", width = 7,
                                          showscale = true,
                                          colorbar = PB.attr(title = "a_y [m/s²]", len = 0.35,
                                                             x = SCENE_RIGHT - 0.02, y = 0.5)),
                           name = "CG path", hovertemplate = "a_y %{line.color:.2f}<extra></extra>"))

    tx = [Float64[] for _ in 1:4]; ty = [Float64[] for _ in 1:4]
    for n in eachindex(M.full.X)
        R = rotz(M.full.psi[n])
        for (q, (xs, ys)) in enumerate(stations(GEOM))
            p = [M.full.X[n], M.full.Y[n], 0.0] + R*[xs, ys, 0.0]
            push!(tx[q], p[1]); push!(ty[q], p[2])
        end
    end
    for q in 1:4
        push!(tr, PB.scatter3d(x = tx[q], y = ty[q], z = fill(0.015, length(tx[q])),
                               mode = "lines", line = PB.attr(color = "rgb(125,125,125)", width = 2),
                               hoverinfo = "skip", showlegend = q == 1, name = "tyre tracks"))
    end

    # 2D time histories. Plotted at full solver resolution, not the animation
    # sampling, so the traces are the signals themselves and not a decimation.
    push!(tr, PB.scatter(x = M.full.t, y = rad2deg.(M.full.delta .* STEER_RATIO),
                         mode = "lines", line = PB.attr(color = "rgb(35,35,40)", width = 2),
                         xaxis = "x", yaxis = "y", showlegend = false,
                         hovertemplate = "t %{x:.2f} s<br>SW %{y:.1f}°<extra></extra>"))
    push!(tr, PB.scatter(x = M.full.t, y = rad2deg.(M.full.psi),
                         mode = "lines", line = PB.attr(color = "rgb(38,80,140)", width = 2),
                         xaxis = "x2", yaxis = "y2", showlegend = false,
                         hovertemplate = "t %{x:.2f} s<br>ψ %{y:.2f}°<extra></extra>"))
    return tr
end

"Moving cursors on the 2D panels, one per animated quantity."
function cursor_traces(M, n)
    return (PB.scatter(x = [M.t[n]], y = [rad2deg(M.delta[n]*STEER_RATIO)], mode = "markers",
                       marker = PB.attr(size = 9, color = "rgb(220,45,45)"),
                       xaxis = "x", yaxis = "y", showlegend = false, hoverinfo = "skip"),
            PB.scatter(x = [M.t[n]], y = [rad2deg(M.psi[n])], mode = "markers",
                       marker = PB.attr(size = 9, color = "rgb(220,45,45)"),
                       xaxis = "x2", yaxis = "y2", showlegend = false, hoverinfo = "skip"))
end

"""
    dynamic_parts(M, n, lo, hi)

Every mesh that moves, as `(mesh, colour)` pairs, in a fixed order: body, cabin,
four wheels, then the steering-wheel gauge (rim, hub, three spokes, marker) and
its fixed reference tick. Initial traces and animation frames both go through
this, so they cannot fall out of step.
"""
function dynamic_parts(M, n, lo, hi)
    meshes, colors = car_meshes(GEOM, M.X[n], M.Y[n], M.psi[n], M.roll[n], M.pitch[n],
                                M.delta[n], [M.Fz[q][n] for q in 1:4], lo, hi)
    parts = Tuple{Any,String}[]
    for (q, mh) in enumerate(meshes)
        push!(parts, (mh, q == 1 ? BODY_COLOR : q == 2 ? CABIN_COLOR : colors[q-2]))
    end
    gc = [M.X[n], M.Y[n], 0.0] .+ SW_OFFSET
    append!(parts, steering_wheel_meshes(gc, M.delta[n]*STEER_RATIO; radius = SW_RADIUS))
    push!(parts, (steering_tick_mesh(gc; radius = SW_RADIUS), "rgb(70,70,75)"))
    return parts
end

function car_traces!(tr, M, n, lo, hi; opacity = 1.0, legend = false)
    for (q, (mh, col)) in enumerate(dynamic_parts(M, n, lo, hi))
        x, y, z, i, j, k = mh
        push!(tr, PB.mesh3d(x = x, y = y, z = z, i = i, j = j, k = k, color = col,
                            flatshading = true, opacity = opacity, showscale = false,
                            hoverinfo = "skip", showlegend = legend && q == 7,
                            name = "steering wheel ($(Int(STEER_RATIO)):1)"))
    end
    return tr
end

frame_title(M, n, vx0) = @sprintf(
    "Double lane change, MPC driver — %.1f m/s (%.0f km/h)  |  t = %5.2f s  |  a_y = %6.2f m/s²  |  steering wheel %+6.1f°  (road wheel %+5.2f°)  |  β = %+.2f°",
    vx0, 3.6*vx0, M.t[n], M.ay[n], rad2deg(M.delta[n]*STEER_RATIO),
    rad2deg(M.delta[n]), rad2deg(M.beta[n]))

function scene_layout(M, course; follow, animated, title)
    scene = if follow
        PB.attr(xaxis = PB.attr(range = [M.X[1]-WINDOW, M.X[1]+WINDOW], title = "X [m]"),
                yaxis = PB.attr(range = [M.Y[1]-WINDOW, M.Y[1]+WINDOW], title = "Y [m]"),
                zaxis = PB.attr(range = [Z_RANGE[1], Z_RANGE[2]], title = ""),
                aspectmode = "manual",
                aspectratio = PB.attr(x = 1, y = 1, z = (Z_RANGE[2]-Z_RANGE[1])/(2WINDOW)),
                camera = PB.attr(eye = PB.attr(x = -1.3, y = -1.2, z = 0.8)),
                domain = PB.attr(x = [0.0, SCENE_RIGHT], y = [0.0, 1.0]))
    else
        PB.attr(xaxis = PB.attr(title = "X [m]"), yaxis = PB.attr(title = "Y [m]"),
                zaxis = PB.attr(title = "", showticklabels = false),
                aspectmode = "data",
                camera = PB.attr(eye = PB.attr(x = -0.9, y = -1.5, z = 1.0)),
                domain = PB.attr(x = [0.0, SCENE_RIGHT], y = [0.0, 1.0]))
    end

    tmax = M.full.t[end]
    layout = PB.Layout(title = PB.attr(text = title, x = 0.02, xanchor = "left",
                                       font = PB.attr(size = 12)),
                       scene = scene, margin = PB.attr(l = 0, r = 0, t = 46, b = 0),
                       paper_bgcolor = "white", legend = PB.attr(x = 0.01, y = 0.97),
                       # 2D panels: steering wheel angle (top) and yaw angle (bottom)
                       xaxis  = PB.attr(domain = PANEL_X, anchor = "y",
                                        range = [0.0, tmax], title = "t [s]",
                                        showgrid = true, gridcolor = "rgb(232,232,232)"),
                       yaxis  = PB.attr(domain = PANEL_Y_SW, anchor = "x",
                                        title = "steering wheel [deg]", zeroline = true,
                                        zerolinecolor = "rgb(170,170,170)",
                                        showgrid = true, gridcolor = "rgb(232,232,232)"),
                       xaxis2 = PB.attr(domain = PANEL_X, anchor = "y2",
                                        range = [0.0, tmax], title = "t [s]",
                                        showgrid = true, gridcolor = "rgb(232,232,232)"),
                       yaxis2 = PB.attr(domain = PANEL_Y_PSI, anchor = "x2",
                                        title = "yaw angle ψ [deg]", zeroline = true,
                                        zerolinecolor = "rgb(170,170,170)",
                                        showgrid = true, gridcolor = "rgb(232,232,232)"),
                       plot_bgcolor = "white")
    animated || return layout

    steps = [PB.attr(method = "animate", label = string(round(M.t[n], digits = 2)),
                     args = [[string(n)], PB.attr(mode = "immediate",
                                                  frame = PB.attr(duration = 40, redraw = true),
                                                  transition = PB.attr(duration = 0))])
             for n in eachindex(M.t)]
    layout[:updatemenus] = [PB.attr(type = "buttons", showactive = false, x = 0.02, y = 0.02,
        xanchor = "left", yanchor = "bottom", direction = "right",
        buttons = [PB.attr(label = "▶ play", method = "animate",
                           args = [nothing, PB.attr(mode = "immediate", fromcurrent = true,
                                                    frame = PB.attr(duration = 40, redraw = true),
                                                    transition = PB.attr(duration = 0))]),
                   PB.attr(label = "❚❚ pause", method = "animate",
                           args = [[nothing], PB.attr(mode = "immediate",
                                                      frame = PB.attr(duration = 0, redraw = false),
                                                      transition = PB.attr(duration = 0))])])]
    layout[:sliders] = [PB.attr(active = 0, y = 0.02, x = 0.18, len = 0.78,
                                pad = PB.attr(t = 0, b = 10),
                                currentvalue = PB.attr(prefix = "t = ", suffix = " s",
                                                       font = PB.attr(size = 13)),
                                steps = steps)]
    return layout
end

"Animated scene with the camera tracking the car."
function build_follow(M, course, path, cones, hit, vx0)
    lo, hi = extrema(reduce(vcat, M.Fz))
    tr = static_traces(M, course, path, cones, hit)
    nstatic = length(tr)
    car_traces!(tr, M, 1, lo, hi; legend = true)
    for c in cursor_traces(M, 1); push!(tr, c); end
    dyn = collect(nstatic : length(tr) - 1)

    frames = PB.PlotlyFrame[]
    for n in eachindex(M.t)
        data = Any[]
        for (mh, col) in dynamic_parts(M, n, lo, hi)
            x, y, z, _, _, _ = mh
            push!(data, PB.attr(x = x, y = y, z = z, color = col))
        end
        push!(data, PB.attr(x = [M.t[n]], y = [rad2deg(M.delta[n]*STEER_RATIO)]))
        push!(data, PB.attr(x = [M.t[n]], y = [rad2deg(M.psi[n])]))
        lay = PB.attr(title = PB.attr(text = frame_title(M, n, vx0)))
        # the domain is restated here: a frame layout update replaces the keys it
        # names, and losing it would let the 3D scene expand over the 2D panels
        lay[:scene] = PB.attr(xaxis = PB.attr(range = [M.X[n]-WINDOW, M.X[n]+WINDOW]),
                              yaxis = PB.attr(range = [M.Y[n]-WINDOW, M.Y[n]+WINDOW]),
                              zaxis = PB.attr(range = [Z_RANGE[1], Z_RANGE[2]]),
                              domain = PB.attr(x = [0.0, SCENE_RIGHT], y = [0.0, 1.0]))
        push!(frames, PB.frame(name = string(n), data = data, traces = dyn, layout = lay))
    end
    return PB.Plot(tr, scene_layout(M, course; follow = true, animated = true,
                                    title = frame_title(M, 1, vx0)), frames)
end

"Static overview of the whole course with ghost cars."
function build_overview(M, course, path, cones, hit, vx0; nghosts = 12)
    lo, hi = extrema(reduce(vcat, M.Fz))
    tr = static_traces(M, course, path, cones, hit)
    for (gi, n) in enumerate(round.(Int, range(1, length(M.t); length = nghosts)))
        car_traces!(tr, M, n, lo, hi; opacity = 0.35 + 0.65*(gi-1)/(nghosts-1))
    end
    ttl = @sprintf("Double lane change, MPC driver — %.1f m/s (%.0f km/h)  |  peak a_y %.2f m/s²  |  cones struck: %d",
                   vx0, 3.6*vx0, maximum(abs, M.ay), count(hit))
    return PB.Plot(tr, scene_layout(M, course; follow = false, animated = false, title = ttl))
end

# ===========================================================================

function local_plotlyjs()
    art = joinpath(dirname(Sys.BINDIR), "share", "julia", "artifacts")
    isdir(art) || return nothing
    for d in readdir(art; join = true)
        f = joinpath(d, "plotly.js"); isfile(f) && return f
    end
    return nothing
end

function write_html(p, path; offline = true)
    mode = "cdn"
    if offline
        src = local_plotlyjs()
        if src === nothing
            @warn "no local plotly.js in the runtime artifacts; falling back to CDN"
        else
            dest = joinpath(dirname(path), "plotly.min.js")
            isfile(dest) || cp(src, dest)
            mode = "directory"
        end
    end
    open(path, "w") do io
        PB.to_html(io, p; include_plotlyjs = mode, full_html = true, autoplay = false)
    end
    return path
end

function main(; course = DLCCourse(), mpc = MPCParams(), offline = true, fail_scene = true)
    P = build_plant(:doubletrack)
    @info "searching for the max pass speed"
    best = best_inset(P; course, mpc)
    @info "limit found" speed_mps=round(best.speed, digits=2) kmh=round(3.6*best.speed) inset=best.inset

    cones = cone_positions(course)
    out = String[]

    M = sample_dlc(; vx0 = best.speed, course, path = best.path, mpc)
    hit = struck(cones, M.full)
    @info "pass run" cones_struck=count(hit) peak_ay=round(maximum(abs, M.ay), digits=2)
    push!(out, write_html(build_follow(M, course, best.path, cones, hit, best.speed),
                          OUT_FOLLOW; offline))
    push!(out, write_html(build_overview(M, course, best.path, cones, hit, best.speed),
                          OUT_OVERVIEW; offline))

    if fail_scene
        vfail = best.speed + 1.5
        Mf = sample_dlc(; vx0 = vfail, course, path = best.path, mpc)
        hitf = struck(cones, Mf.full)
        @info "fail run" speed_mps=round(vfail, digits=2) cones_struck=count(hitf)
        push!(out, write_html(build_follow(Mf, course, best.path, cones, hitf, vfail),
                              OUT_FAIL; offline))
    end

    for f in out
        @info "wrote" file=f size_mb=round(filesize(f)/1024/1024, digits=2)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
