#
# visualize_vehicle_3d.jl
#
# Interactive 3D visualisation of a 3-DOF vehicle and its path.
#
# Run from the package root:
#     julia --project=. scripts/visualize_vehicle_3d.jl
#
# Produces two self-contained HTML files in this directory (drag into a browser,
# left-drag to orbit, scroll to zoom, right-drag to pan):
#
#   vehicle_3d_follow.html    animated, camera tracking the vehicle
#   vehicle_3d_overview.html  whole manoeuvre at once, ghost cars along the path
#
# Rendering uses PlotlyBase, which ships with the Dyad runtime, so there is
# nothing to install.
#
# ---------------------------------------------------------------------------
# A note on what is and is not simulated
#
# The model has three degrees of freedom: vx, vy and yaw rate. It has no roll or
# pitch DOF. The body lean you see is a DISPLAY-ONLY attitude reconstructed from
# ax_lag and ay_lag — the same suspension-lagged accelerations that drive the
# load transfer in the model — scaled by ROLL_DEG_PER_G / PITCH_DEG_PER_G. It is
# consistent with the simulated weight transfer and it makes the animation
# readable, but it is not an independent state. Set the gains to zero to remove
# it entirely.
#
# Everything else on screen is simulated output: CG path, heading, steer angle,
# per-corner vertical load (wheel colour) and the four tyre contact tracks.
# ---------------------------------------------------------------------------

using TestVehicleDynamics
using DyadInterface
using ModelingToolkit
using OrdinaryDiffEqDefault
using PlotlyBase
const PB = PlotlyBase

# Mesh primitives and the car body live in a shared file so this visualiser and
# visualize_dlc_3d.jl cannot drift apart.
include(joinpath(@__DIR__, "vehicle_3d_geometry.jl"))

# --- manoeuvre -------------------------------------------------------------
const DELTA   = 0.08       # held road-wheel steer angle [rad]
const V0      = 25.0       # initial speed [m/s]
const TSPAN   = 6.0        # simulated duration [s]
const NFRAMES = 150        # animation frames

const GEOM = VehicleGeom()

# --- display-only attitude -------------------------------------------------
const ROLL_DEG_PER_G  = 4.5
const PITCH_DEG_PER_G = 1.8

# --- follow camera ---------------------------------------------------------
const WINDOW  = 16.0       # half-width of the tracking window [m]
const Z_RANGE = (0.0, 3.2)

const OUT_FOLLOW   = joinpath(@__DIR__, "vehicle_3d_follow.html")
const OUT_OVERVIEW = joinpath(@__DIR__, "vehicle_3d_overview.html")

# ===========================================================================
# Simulation and sampling
# ===========================================================================

"Simulate the double-track step steer and sample everything the scene needs."
function sample_motion(; delta = DELTA, v0 = V0, T = TSPAN, nframes = NFRAMES)
    sys = DyadInterface.simplify_model(
        TestVehicleDynamics.TestDoubleTrackStepSteer(; name = :dt4))
    sol = solve(ODEProblem(sys, Pair{Any,Any}[sys.delta_max => delta, sys.v0 => v0],
                           (0.0, T)); abstol = 1e-10, reltol = 1e-10)
    sol.retcode == ReturnCode.Success ||
        error("manoeuvre left the model's validity envelope at t = $(sol.t[end])")

    ts = collect(range(0, T; length = nframes))
    g  = 9.81
    grab(sym) = [sol(t; idxs = sym) for t in ts]

    corner_syms = (sys.car.fl.Fz, sys.car.fr.Fz, sys.car.rl.Fz, sys.car.rr.Fz)
    return (; ts,
            X     = grab(sys.car.X),
            Y     = grab(sys.car.Y),
            psi   = grab(sys.car.psi),
            vx    = grab(sys.car.vx),
            vy    = grab(sys.car.vy),
            r     = grab(sys.car.r),
            ay    = grab(sys.car.ay),
            beta  = grab(sys.car.beta),
            # display-only attitude from the suspension-lagged accelerations
            roll  = deg2rad(ROLL_DEG_PER_G)  .* grab(sys.car.ay_lag) ./ g,
            pitch = -deg2rad(PITCH_DEG_PER_G) .* grab(sys.car.ax_lag) ./ g,
            # actual steer applied to the model, not a re-derived ramp
            delta = grab(sys.steer.y),
            Fz    = [grab(s) for s in corner_syms])
end

"World-frame contact point of each wheel station over the whole manoeuvre."
function contact_tracks(M)
    tracks = [(Float64[], Float64[]) for _ in 1:4]
    for n in eachindex(M.ts)
        R = rotz(M.psi[n])
        for (q, (xs, ys)) in enumerate(stations(GEOM))
            p = [M.X[n], M.Y[n], 0.0] + R * [xs, ys, 0.0]
            push!(tracks[q][1], p[1]); push!(tracks[q][2], p[2])
        end
    end
    return tracks
end

# ===========================================================================
# Car traces
# ===========================================================================

"Mesh data for the body shell, cabin and four wheels at frame `n`."
function car_meshes(M, n, Fz_lo, Fz_hi)
    return car_meshes(GEOM, M.X[n], M.Y[n], M.psi[n], M.roll[n], M.pitch[n],
                      M.delta[n], [M.Fz[q][n] for q in 1:4], Fz_lo, Fz_hi)
end

"Velocity vector at the CG, drawn in the ground plane."
function velocity_trace(M, n; scale = 0.35)
    R  = rotz(M.psi[n])
    v  = R * [M.vx[n], M.vy[n], 0.0] .* scale
    p0 = [M.X[n], M.Y[n], 0.05]
    p1 = p0 + v
    return [p0[1], p1[1]], [p0[2], p1[2]], [p0[3], p1[3]]
end

# ===========================================================================
# Scene assembly
# ===========================================================================

const BODY_COLOR  = "rgb(38,80,140)"
const CABIN_COLOR = "rgb(150,185,225)"

"Static traces: ground, grid, CG path coloured by ay, four tyre tracks."
function static_traces(M)
    traces = PB.AbstractTrace[]

    xlo, xhi = extrema(M.X); ylo, yhi = extrema(M.Y)
    pad = 25.0
    xlo -= pad; xhi += pad; ylo -= pad; yhi += pad

    push!(traces, PB.mesh3d(x = [xlo, xhi, xhi, xlo], y = [ylo, ylo, yhi, yhi],
                            z = fill(-0.02, 4), i = [0, 0], j = [1, 2], k = [2, 3],
                            color = "rgb(238,238,236)", opacity = 1.0,
                            flatshading = true, hoverinfo = "skip",
                            showscale = false, name = "road"))

    gx, gy, gz = Float64[], Float64[], Float64[]
    for xg in ceil(xlo/10)*10 : 10.0 : xhi
        append!(gx, [xg, xg, NaN]); append!(gy, [ylo, yhi, NaN]); append!(gz, [0.0, 0.0, NaN])
    end
    for yg in ceil(ylo/10)*10 : 10.0 : yhi
        append!(gx, [xlo, xhi, NaN]); append!(gy, [yg, yg, NaN]); append!(gz, [0.0, 0.0, NaN])
    end
    push!(traces, PB.scatter3d(x = gx, y = gy, z = gz, mode = "lines",
                               line = PB.attr(color = "rgb(210,210,208)", width = 1),
                               hoverinfo = "skip", showlegend = false, name = "10 m grid"))

    for (q, (tx, ty)) in enumerate(contact_tracks(M))
        push!(traces, PB.scatter3d(x = tx, y = ty, z = fill(0.012, length(tx)),
                                   mode = "lines",
                                   line = PB.attr(color = "rgb(120,120,120)", width = 2),
                                   hoverinfo = "skip", showlegend = q == 1,
                                   name = "tyre tracks"))
    end

    push!(traces, PB.scatter3d(x = M.X, y = M.Y, z = fill(0.05, length(M.X)),
                               mode = "lines",
                               line = PB.attr(color = M.ay, colorscale = "Viridis",
                                              width = 7, showscale = true,
                                              colorbar = PB.attr(title = "a_y [m/s²]",
                                                                 len = 0.45, x = 0.98)),
                               name = "CG path", hovertemplate = "a_y %{line.color:.2f}<extra></extra>"))
    return traces
end

"Build the full figure. `follow = true` gives the tracking camera window."
function build_figure(M; follow::Bool)
    Fz_all = reduce(vcat, M.Fz)
    Fz_lo, Fz_hi = extrema(Fz_all)

    traces   = static_traces(M)
    n_static = length(traces)

    # dynamic traces, in order: body, cabin, FL, FR, RL, RR wheels, velocity vector
    meshes, colors = car_meshes(M, 1, Fz_lo, Fz_hi)
    for (q, mh) in enumerate(meshes)
        x, y, z, i, j, k = mh
        col  = q == 1 ? BODY_COLOR : q == 2 ? CABIN_COLOR : colors[q-2]
        name = q == 1 ? "body" : q == 2 ? "cabin" : ("wheel " * ("FL","FR","RL","RR")[q-2])
        push!(traces, PB.mesh3d(x = x, y = y, z = z, i = i, j = j, k = k,
                                color = col, flatshading = true, opacity = 1.0,
                                showscale = false, hoverinfo = "skip",
                                showlegend = false, name = name))
    end
    vx_, vy_, vz_ = velocity_trace(M, 1)
    push!(traces, PB.scatter3d(x = vx_, y = vy_, z = vz_, mode = "lines",
                               line = PB.attr(color = "rgb(200,40,40)", width = 6),
                               hoverinfo = "skip", name = "velocity"))

    dyn_idx = collect(n_static : length(traces) - 1)   # 0-based for plotly

    frames = PB.PlotlyFrame[]
    for n in eachindex(M.ts)
        meshes, colors = car_meshes(M, n, Fz_lo, Fz_hi)
        data = Any[]
        for (q, mh) in enumerate(meshes)
            x, y, z, _, _, _ = mh
            col = q == 1 ? BODY_COLOR : q == 2 ? CABIN_COLOR : colors[q-2]
            push!(data, PB.attr(x = x, y = y, z = z, color = col))
        end
        vx_, vy_, vz_ = velocity_trace(M, n)
        push!(data, PB.attr(x = vx_, y = vy_, z = vz_))

        lay = PB.attr(title = PB.attr(text = frame_title(M, n)))
        if follow
            lay[:scene] = PB.attr(
                xaxis = PB.attr(range = [M.X[n] - WINDOW, M.X[n] + WINDOW]),
                yaxis = PB.attr(range = [M.Y[n] - WINDOW, M.Y[n] + WINDOW]),
                zaxis = PB.attr(range = [Z_RANGE[1], Z_RANGE[2]]))
        end
        push!(frames, PB.frame(name = string(n), data = data,
                               traces = dyn_idx, layout = lay))
    end

    layout = scene_layout(M; follow = follow)
    return PB.Plot(traces, layout, frames)
end

function frame_title(M, n)
    return "3-DOF double-track vehicle — step steer  " *
           "|  t = $(lpad(string(round(M.ts[n], digits=2)), 5)) s" *
           "  |  v = $(round(hypot(M.vx[n], M.vy[n]), digits=1)) m/s" *
           "  |  a_y = $(round(M.ay[n], digits=2)) m/s²" *
           "  |  r = $(round(M.r[n], digits=3)) rad/s" *
           "  |  β = $(round(rad2deg(M.beta[n]), digits=1))°"
end

"""
    scene_layout(M; follow, animated, title)

`animated` must be `false` for a figure built without frames. Plotly's play
button and slider are `method = "animate"` actions: with no frames attached they
are present but inert, which looks like a broken control.
"""
function scene_layout(M; follow::Bool, animated::Bool = true, title = frame_title(M, 1))
    scene = if follow
        PB.attr(xaxis = PB.attr(range = [M.X[1] - WINDOW, M.X[1] + WINDOW], title = "X [m]"),
                yaxis = PB.attr(range = [M.Y[1] - WINDOW, M.Y[1] + WINDOW], title = "Y [m]"),
                zaxis = PB.attr(range = [Z_RANGE[1], Z_RANGE[2]], title = ""),
                aspectmode = "manual",
                aspectratio = PB.attr(x = 1, y = 1, z = (Z_RANGE[2]-Z_RANGE[1]) / (2WINDOW)),
                camera = PB.attr(eye = PB.attr(x = -1.25, y = -1.25, z = 0.85)))
    else
        PB.attr(xaxis = PB.attr(title = "X [m]"), yaxis = PB.attr(title = "Y [m]"),
                zaxis = PB.attr(title = "", showticklabels = false),
                aspectmode = "data",
                camera = PB.attr(eye = PB.attr(x = -1.1, y = -1.4, z = 1.1)))
    end

    layout = PB.Layout(
        title = PB.attr(text = title, x = 0.02, xanchor = "left",
                        font = PB.attr(size = 13)),
        scene = scene,
        margin = PB.attr(l = 0, r = 0, t = 46, b = 0),
        paper_bgcolor = "white",
        legend = PB.attr(x = 0.01, y = 0.98))

    animated || return layout   # no frames => no transport controls

    steps = [PB.attr(method = "animate", label = string(round(M.ts[n], digits = 2)),
                     args = [[string(n)], PB.attr(mode = "immediate",
                                                  frame = PB.attr(duration = 40, redraw = true),
                                                  transition = PB.attr(duration = 0))])
             for n in eachindex(M.ts)]

    layout[:updatemenus] = [PB.attr(type = "buttons", showactive = false,
        x = 0.02, y = 0.02, xanchor = "left", yanchor = "bottom", direction = "right",
        buttons = [
            PB.attr(label = "▶ play", method = "animate",
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

"Title for the whole-manoeuvre overview, which shows no single instant."
function overview_title(M)
    return "3-DOF double-track vehicle — step steer, full manoeuvre  " *
           "|  $(round(M.ts[end], digits=1)) s" *
           "  |  peak a_y = $(round(maximum(M.ay), digits=2)) m/s²" *
           "  |  peak r = $(round(maximum(M.r), digits=3)) rad/s" *
           "  |  ghost cars every $(round(M.ts[end]/8, digits=2)) s"
end

"Static overview: full path with ghost cars at regular intervals."
function build_overview(M; nghosts = 9)
    Fz_all = reduce(vcat, M.Fz)
    Fz_lo, Fz_hi = extrema(Fz_all)
    traces = static_traces(M)

    idxs = round.(Int, range(1, length(M.ts); length = nghosts))
    for (gi, n) in enumerate(idxs)
        meshes, colors = car_meshes(M, n, Fz_lo, Fz_hi)
        op = 0.35 + 0.65 * (gi - 1) / (nghosts - 1)    # fades in along the path
        for (q, mh) in enumerate(meshes)
            x, y, z, i, j, k = mh
            col = q == 1 ? BODY_COLOR : q == 2 ? CABIN_COLOR : colors[q-2]
            push!(traces, PB.mesh3d(x = x, y = y, z = z, i = i, j = j, k = k,
                                    color = col, flatshading = true, opacity = op,
                                    showscale = false, hoverinfo = "skip",
                                    showlegend = false))
        end
    end
    return PB.Plot(traces, scene_layout(M; follow = false, animated = false,
                                        title = overview_title(M)))
end

# ===========================================================================

"""
    local_plotlyjs()

Path to the `plotly.js` that ships with the Dyad runtime, or `nothing`. Used by
`offline = true` so the HTML renders without an internet connection.
"""
function local_plotlyjs()
    art = joinpath(dirname(Sys.BINDIR), "share", "julia", "artifacts")
    isdir(art) || return nothing
    for d in readdir(art; join = true)
        f = joinpath(d, "plotly.js")
        isfile(f) && return f
    end
    return nothing
end

"""
    write_html(p, path; offline = false)

`offline = false` links plotly.js from the CDN — small files, needs a
connection. `offline = true` copies the runtime's plotly.js next to the HTML as
`plotly.min.js` and links that instead, so the scenes work with no network.
(PlotlyBase's inline-a-local-file branch references an undefined variable, hence
the sidecar rather than a single self-contained file.)
"""
function write_html(p, path; offline = false)
    mode = "cdn"
    if offline
        src = local_plotlyjs()
        if src === nothing
            @warn "no local plotly.js found in the runtime artifacts; falling back to CDN"
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

function main(; offline = false)
    @info "Simulating"
    M = sample_motion()
    @info "Manoeuvre" duration_s=M.ts[end] path_length_m=round(
        sum(hypot(M.X[n+1]-M.X[n], M.Y[n+1]-M.Y[n]) for n in 1:length(M.ts)-1), digits=1) frames=length(M.ts)

    @info "Building animated follow-camera scene"
    write_html(build_figure(M; follow = true), OUT_FOLLOW; offline = offline)
    @info "Building static overview scene"
    write_html(build_overview(M), OUT_OVERVIEW; offline = offline)

    for f in (OUT_FOLLOW, OUT_OVERVIEW)
        @info "wrote" file=f size_kb=round(filesize(f)/1024, digits=1)
    end
    return OUT_FOLLOW, OUT_OVERVIEW
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
