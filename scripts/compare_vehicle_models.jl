#
# compare_vehicle_models.jl
#
# Regenerates the 3-DOF single-track (bicycle) vs double-track comparison figure.
#
# Run from the package root:
#     julia --project=. scripts/compare_vehicle_models.jl
#
# Both vehicles extend the same PlanarVehicleBase and instantiate the same
# WheelStation, so the body equations of motion and the corner tyre physics are
# identical between them. The only structural difference is the number of wheel
# stations and, with it, the presence of lateral load transfer and a track-width
# yaw moment. Everything the two curves do differently traces back to that.
#

using TestVehicleDynamics
using DyadInterface
using ModelingToolkit
using OrdinaryDiffEqDefault
using Plots, Measures

const OUTFILE = joinpath(@__DIR__, "compare_vehicle_models.png")

# Steer angles used for the two time-history panels [rad]
const DELTA_SUBLIMIT = 0.06
const DELTA_LIMIT    = 0.08
# Time at which the steady-state values for the handling diagram are sampled [s]
const T_SAMPLE       = 5.5
# |beta| above this is a spin, not a steady cornering state [deg]
const BETA_SPIN      = 15.0

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

"Build and structurally simplify both step-steer harnesses."
function build_models()
    bic = DyadInterface.simplify_model(TestVehicleDynamics.TestBicycleStepSteer(; name = :bic))
    dt4 = DyadInterface.simplify_model(TestVehicleDynamics.TestDoubleTrackStepSteer(; name = :dt4))
    return bic, dt4
end

"""
    step_steer(sys; delta, v0, T, extra)

Run one step-steer manoeuvre. `delta` is the road-wheel angle held after the
0.15 s ramp [rad], `v0` the initial and nominally held speed [m/s]. `extra`
takes additional symbolic parameter overrides, e.g. `[sys.car.eps_f => 0.45]`.
"""
function step_steer(sys; delta = DELTA_SUBLIMIT, v0 = 25.0, T = 6.0,
                    extra = Pair{Any,Any}[])
    p = Pair{Any,Any}[sys.delta_max => delta, sys.v0 => v0]
    append!(p, extra)
    solve(ODEProblem(sys, p, (0.0, T)); abstol = 1e-10, reltol = 1e-10)
end

"""
    steady(sys, delta; front_alpha, rear_alpha)

Sample the quasi-steady cornering state at `T_SAMPLE`. Returns `ok = false` when
the run terminated early — the base model asserts `vx > -v_eps`, because the
regularised slip-angle definition is only valid in forward motion.
"""
function steady(sys, delta; front_alpha, rear_alpha, extra = Pair{Any,Any}[])
    bad = (ok = false, r = NaN, ay = NaN, beta = NaN, vx = NaN, af = NaN, ar = NaN)
    s = step_steer(sys; delta = delta, extra = extra)
    (s.retcode == ReturnCode.Success && s.t[end] >= T_SAMPLE) || return bad
    return (ok   = true,
            r    = s(T_SAMPLE, idxs = sys.car.r),
            ay   = s(T_SAMPLE, idxs = sys.car.ay),
            beta = rad2deg(s(T_SAMPLE, idxs = sys.car.beta)),
            vx   = s(T_SAMPLE, idxs = sys.car.vx),
            af   = rad2deg(s(T_SAMPLE, idxs = front_alpha)),
            ar   = rad2deg(s(T_SAMPLE, idxs = rear_alpha)))
end

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

"Yaw-rate time history of both models at a single steer angle."
function panel_yaw(bic, dt4, delta; title, legend, tt)
    sb = step_steer(bic; delta = delta)
    sd = step_steer(dt4; delta = delta)
    p = plot(; title = title, xlabel = "t [s]", ylabel = "r [rad/s]", legend = legend)
    plot!(p, tt, sb.(tt, idxs = bic.car.r), lw = 2.5, label = "bicycle (2 tyres)")
    plot!(p, tt, sd.(tt, idxs = dt4.car.r), lw = 2.5, ls = :dash, label = "double-track (4 tyres)")
    return p
end

"Steady-state lateral acceleration against steer angle, spun points discarded."
function panel_handling(bic, dt4; deltas = range(0.002, 0.075, length = 40))
    degb, ayb = Float64[], Float64[]
    degd, ayd = Float64[], Float64[]
    for d in deltas
        B = steady(bic, d; front_alpha = bic.car.front.alpha, rear_alpha = bic.car.rear.alpha)
        D = steady(dt4, d; front_alpha = dt4.car.fl.alpha,    rear_alpha = dt4.car.rl.alpha)
        if B.ok && abs(B.beta) < BETA_SPIN
            push!(degb, rad2deg(d)); push!(ayb, B.ay)
        end
        if D.ok && abs(D.beta) < BETA_SPIN
            push!(degd, rad2deg(d)); push!(ayd, D.ay)
        end
    end
    p = plot(degb, ayb; lw = 2.5, label = "bicycle",
             title = "Handling diagram (steady state, 25 m/s)",
             xlabel = "steer angle [deg]", ylabel = "lateral acceleration ay [m/s²]",
             legend = :bottomright)
    plot!(p, degd, ayd, lw = 2.5, ls = :dash, label = "double-track")
    isempty(degb) || scatter!(p, [degb[end]], [ayb[end]], ms = 6, mc = :red,
                              label = "bicycle spins beyond here")
    return p
end

"Double-track vertical corner loads through the manoeuvre."
function panel_corner_loads(dt4, delta; tt)
    s = step_steer(dt4; delta = delta)
    p = plot(; title = "Double-track corner loads (δ=$(round(rad2deg(delta), digits=2))°)",
             xlabel = "t [s]", ylabel = "Fz [N]", legend = :right)
    for (sym, lab) in ((dt4.car.fl.Fz, "front left (inner)"),
                       (dt4.car.fr.Fz, "front right (outer)"),
                       (dt4.car.rl.Fz, "rear left (inner)"),
                       (dt4.car.rr.Fz, "rear right (outer)"))
        plot!(p, tt, s.(tt, idxs = sym), lw = 2.5, label = lab)
    end
    mean_corner = s(0.0, idxs = dt4.car.fl.Fz + dt4.car.fr.Fz +
                                dt4.car.rl.Fz + dt4.car.rr.Fz) / 4
    hline!(p, [mean_corner], lc = :gray, ls = :dot, lw = 1.5, label = "mean corner load")
    return p
end

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

"Print the steady-state comparison table that backs the handling diagram."
function print_table(bic, dt4)
    fmt(x, n = 3) = isnan(x) ? "  --" : string(round(x, digits = n))
    println("\nSteady-state step-steer comparison at ", T_SAMPLE, " s, 25 m/s")
    println(rpad("δ[deg]", 8), "| ", rpad("BIC  r", 10), rpad("ay", 8), rpad("β", 9),
            rpad("αf-αr", 9), "| ", rpad("DT   r", 10), rpad("ay", 8), rpad("β", 9), "αf-αr")
    for d in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
        B = steady(bic, d; front_alpha = bic.car.front.alpha, rear_alpha = bic.car.rear.alpha)
        D = steady(dt4, d; front_alpha = dt4.car.fl.alpha,    rear_alpha = dt4.car.rl.alpha)
        tag(S) = !S.ok ? "INVALID" : (abs(S.beta) > BETA_SPIN ? "SPUN" : fmt(S.af - S.ar, 2))
        println(rpad(round(rad2deg(d), digits = 2), 8), "| ",
                rpad(fmt(B.r, 4), 10), rpad(fmt(B.ay), 8), rpad(fmt(B.beta, 2), 9), rpad(tag(B), 9), "| ",
                rpad(fmt(D.r, 4), 10), rpad(fmt(D.ay), 8), rpad(fmt(D.beta, 2), 9), tag(D))
    end
    println()
end

# ---------------------------------------------------------------------------

function main()
    @info "Building and simplifying both vehicle models"
    bic, dt4 = build_models()
    @info "States" bicycle = length(unknowns(bic)) double_track = length(unknowns(dt4))

    tt = range(0, 6, length = 1200)
    gr(size = (1300, 950))

    @info "Running manoeuvres"
    p1 = panel_yaw(bic, dt4, DELTA_SUBLIMIT;
                   title = "Yaw rate — sub-limit step steer (δ=$(round(rad2deg(DELTA_SUBLIMIT), digits=2))°)",
                   legend = :bottomright, tt = tt)
    p2 = panel_yaw(bic, dt4, DELTA_LIMIT;
                   title = "Yaw rate — limit step steer (δ=$(round(rad2deg(DELTA_LIMIT), digits=2))°)",
                   legend = :topleft, tt = tt)
    annotate!(p2, 4.3, 0.30, text("bicycle diverges\n(rear saturates → spin)", 8, :left, :steelblue))

    p3 = panel_handling(bic, dt4)
    annotate!(p3, 2.6, 5.2, text("double-track loses ~6% ay:\nroll transfer on a\ndegressive tyre",
                                 8, :left, :firebrick))

    p4 = panel_corner_loads(dt4, DELTA_LIMIT; tt = tt)

    plot(p1, p2, p3, p4; layout = (2, 2),
         left_margin = 12mm, bottom_margin = 6mm, top_margin = 3mm, right_margin = 4mm)
    savefig(OUTFILE)
    @info "Figure written" OUTFILE

    print_table(bic, dt4)
    return OUTFILE
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
