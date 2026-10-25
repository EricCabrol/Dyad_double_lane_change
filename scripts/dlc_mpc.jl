#
# dlc_mpc.jl
#
# Closed-loop double lane change with an MPC driver, and a search for the
# highest entry speed at which the manoeuvre is completed inside the gates.
#
# Run from the package root:
#     julia --project=. scripts/dlc_mpc.jl
#
# ---------------------------------------------------------------------------
# Structure
#
#   Dyad  : the plant only — DLCPlantDoubleTrack / DLCPlantBicycle, each a
#           3-DOF vehicle plus a speed-hold PI and a first-order steering
#           actuator. `act.delta_cmd` is a free parameter with no binding, so
#           it can be written at every control instant; the actuator lag then
#           supplies the hold dynamics a real steering system would.
#
#   Julia : the reference path, the MPC, and the scoring. A QP solver cannot be
#           written as Dyad relations, and the plant does not need the path —
#           only the controller and the pass/fail scorer do.
#
# Protocol: constant speed. The speed-hold PI fights cornering drag so the
# manoeuvre is a pure lateral-grip limit, with no longitudinal load transfer
# helping the front axle. This is harsher than the ISO coast-down protocol.
# ---------------------------------------------------------------------------

using TestVehicleDynamics
using DyadInterface
using ModelingToolkit
using OrdinaryDiffEqDefault
using LinearAlgebra
using Printf
using Plots, Measures

# ===========================================================================
# Reference path
# ===========================================================================

"""
Double lane change course geometry.

Section lengths, gate widths and the lateral offset are all parameters rather
than hard-coded to a standard. The defaults approximate ISO 3888-1. Gate widths
follow the usual `k·w + 0.25` form.

ISO 3888-2 (the moose test) is reachable by setting
`L1=15, W1=13.5, L2=11, W2=12.5, L3=12, k3=1.0`, but note that for a 1.76 m wide
car those corridors leave the centre-of-gravity a *negative* margin in gate 3 —
the body cannot fit through with any realistic yaw angle, so no controller will
pass it. That is a property of the geometry, not of the driver.
"""
Base.@kwdef struct DLCCourse
    L1::Float64 = 15.0     # entry gate
    W1::Float64 = 30.0     # first transition (no cones)
    L2::Float64 = 25.0     # side lane gate
    W2::Float64 = 25.0     # second transition (no cones)
    L3::Float64 = 30.0     # exit gate
    D::Float64  = 3.5      # lateral offset
    w::Float64  = 1.76     # vehicle width
    k1::Float64 = 1.1
    k3::Float64 = 1.2
    k5::Float64 = 1.3
end

total_length(c::DLCCourse) = c.L1 + c.W1 + c.L2 + c.W2 + c.L3

"A cone gate: vehicle corners inside [x0,x1] must satisfy |Y - yc| <= half."
struct Gate
    x0::Float64; x1::Float64; yc::Float64; half::Float64
end

"True cone gates, used for scoring. Half-widths are the physical corridor."
function gates(c::DLCCourse)
    x3 = c.L1 + c.W1
    x5 = x3 + c.L2 + c.W2
    hw(k) = (k*c.w + 0.25)/2
    return (Gate(0.0, c.L1,      0.0, hw(c.k1)),
            Gate(x3,  x3 + c.L2, c.D, hw(c.k3)),
            Gate(x5,  x5 + c.L3, 0.0, hw(c.k5)))
end

# ---------------------------------------------------------------------------
# Reference path
# ---------------------------------------------------------------------------

"Sampled reference path with linear interpolation on a uniform station grid."
struct RefPath
    ds::Float64
    y::Vector{Float64}
    psi::Vector{Float64}
    kap::Vector{Float64}
    s_end::Float64
end

function _interp(v::Vector{Float64}, ds, s)
    x = s/ds + 1
    i = clamp(floor(Int, x), 1, length(v) - 1)
    f = clamp(x - i, 0.0, 1.0)
    return (1 - f)*v[i] + f*v[i+1]
end
y_ref(p::RefPath, s)     = _interp(p.y,   p.ds, s)
psi_ref(p::RefPath, s)   = _interp(p.psi, p.ds, s)
kappa_ref(p::RefPath, s) = _interp(p.kap, p.ds, s)

"""
    reference(course; yaw_allow, ds, runout)

Minimum-curvature path through the gate corridors.

A centreline that snaps from lane to lane is *not* the path a driver takes: the
gates are wider than the car, so the corridor is used and corners are cut. For
this course that distinction is not cosmetic — the smoothstep centreline needs
R ≈ 44 m where the corner-cutting path needs R ≈ 115 m, a factor 2.6 in required
lateral acceleration, which moves the achievable speed by about 1.6x.

Minimising the discrete `∑(y″)²` subject to corridor boxes is a box-constrained
QP; it is solved exactly by an active-set loop (solve equality-constrained,
pin the worst violator, repeat). The result is spline-like, so curvature is
continuous and the controller is never asked for a step input.

`yaw_allow` shrinks the corridor for the planner to leave room for body yaw. It
only shapes the reference — pass/fail is always judged on the true corner
positions against the true gates.
"""
function reference(c::DLCCourse; yaw_allow = 0.05, ds = 0.5, runout = 45.0)
    s = collect(0.0:ds:(total_length(c) + runout))
    n = length(s)
    gs = gates(c)

    lo = fill(-Inf, n); hi = fill(Inf, n)
    for (i, si) in enumerate(s), (gi, g) in enumerate(gs)
        # the exit gate corridor is extended through the run-out: after the
        # course you keep driving straight in that lane, and it stops the
        # unconstrained tail from drifting off on a residual slope
        x1 = gi == length(gs) ? s[end] : g.x1
        if g.x0 <= si <= x1
            h = g.half - c.w/2 - yaw_allow
            lo[i] = max(lo[i], g.yc - h); hi[i] = min(hi[i], g.yc + h)
        end
    end

    # An inset larger than (gate half-width - w/2) inverts a corridor. The QP
    # then has no feasible point and silently returns a kinked path with absurd
    # curvature, so check rather than trusting the result.
    bad = findfirst(i -> lo[i] > hi[i], 1:n)
    bad === nothing || error(
        "corridor inverted at s = $(s[bad]) m: yaw_allow = $yaw_allow exceeds the " *
        "available margin. Max usable inset for this course is " *
        "$(round(minimum(g.half for g in gs) - c.w/2, digits=3)) m.")

    D2 = zeros(n-2, n)
    for i in 1:n-2; D2[i,i] = 1.0; D2[i,i+1] = -2.0; D2[i,i+2] = 1.0; end

    fixed = Dict(1 => 0.0, 2 => 0.0)      # start straight at Y = 0
    y = zeros(n)
    for _ in 1:4n
        fi = sort(collect(keys(fixed))); fr = setdiff(1:n, fi)
        Af = D2[:, fr]; Ac = D2[:, fi]; yc = [fixed[i] for i in fi]
        y[fr] = (Af'Af) \ (-(Af'Ac*yc)); y[fi] = yc
        worst, wi, wv = 1e-9, 0, 0.0
        for i in fr
            v = y[i] > hi[i] ? y[i] - hi[i] : y[i] < lo[i] ? lo[i] - y[i] : 0.0
            if v > worst; worst, wi, wv = v, i, (y[i] > hi[i] ? hi[i] : lo[i]); end
        end
        wi == 0 && break
        fixed[wi] = wv
    end

    dy  = [i == 1 ? (y[2]-y[1])/ds : i == n ? (y[n]-y[n-1])/ds : (y[i+1]-y[i-1])/(2ds) for i in 1:n]
    ddy = [i == 1 || i == n ? 0.0 : (y[i-1] - 2y[i] + y[i+1])/ds^2 for i in 1:n]
    return RefPath(ds, y, atan.(dy), ddy ./ (1 .+ dy.^2).^1.5, total_length(c))
end

"World-frame (x,y) of the four body corners."
function corners(X, Y, psi; len = 4.40, wid = 1.76, x_off = -0.10)
    c, s = cos(psi), sin(psi)
    out = NTuple{2,Float64}[]
    for sx in (-1, 1), sy in (-1, 1)
        bx = x_off + sx*len/2
        by = sy*wid/2
        push!(out, (X + c*bx - s*by, Y + s*bx + c*by))
    end
    return out
end

"""
    score(traj, gs)

Worst corridor margin over the run, in metres. Positive means every corner
stayed inside every gate it passed through; negative is the deepest incursion.
"""
function score(traj, gs)
    worst = Inf
    for n in eachindex(traj.t)
        for (cx, cy) in corners(traj.X[n], traj.Y[n], traj.psi[n])
            for g in gs
                (g.x0 <= cx <= g.x1) || continue
                worst = min(worst, g.half - abs(cy - g.yc))
            end
        end
    end
    return worst
end

# ===========================================================================
# MPC
# ===========================================================================

"Linear prediction model parameters. Cornering stiffnesses are per axle."
Base.@kwdef struct VehicleLin
    m::Float64   = 1600.0
    Iz::Float64  = 2400.0
    a::Float64   = 1.2
    b::Float64   = 1.4
    Cf::Float64  = 126983.0
    Cr::Float64  = 139230.0
    tau_s::Float64 = 0.06
end

"""
MPC tuning.

Two separate things are being traded here, and they are not equally sensitive:

  * **Achievable speed** barely moves. Over a 16x range of `q_ey` and a 6x range
    of `S` the bisected max speed changes by under 3 % (see `tune_driver`), so
    the headline number is a property of the vehicle and the course.

  * **Steering effort** moves a great deal, and an earlier version of this file
    was tuned only on speed, which cannot see it. With `R = 0.5` a 0.1 m lateral
    error costs 0.48 while a 0.1 rad steering input costs 0.005 — steering is
    ~100x cheaper than error, so the optimum spent +-110 deg of wheel chasing
    millimetres. That was the true optimum of a badly scaled cost, not a solver
    fault: the condensing is exact and the result is unchanged out to a 7 s
    horizon.

`R = 50` balances a 0.1 m error against roughly a 2.5 deg road-wheel input. It
halves peak steering and rms steering rate at no cost in speed.

**Horizon.** `N = 100` is 2.0 s, which is 65 m of preview at the limit speed —
enough to see the next gate while still completing the current transition. It
was chosen by convergence of the applied first move `U[1]`, which is the only
part of the solution that ever reaches the plant:

    N = 25 (0.5 s)   U1 off by up to 250 %
    N = 50 (1.0 s)   U1 off by up to 5.5 %,  costs 0.9 % of max speed
    N = 75 (1.5 s)   U1 within 0.5 %
    N = 100 (2.0 s)  converged;  max speed within 0.1 % of N = 150

Note this depends on the cost scaling: with the old aggressive `R = 0.5` the
closed loop was fast enough that even 1.0 s looked converged. Softening `R`
slowed the loop and lengthened the horizon it needs, so the two must be
re-checked together.
"""
Base.@kwdef struct MPCParams
    N::Int       = 100      # horizon steps
    Ts::Float64  = 0.02     # control period [s] -> 2.0 s preview
    q_ey::Float64    = 48.0
    q_epsi::Float64  = 6.0
    q_r::Float64     = 0.30
    R::Float64       = 50.0    # on delta
    S::Float64       = 300.0   # on delta rate
    delta_max::Float64 = 0.55
end

"""
Continuous error dynamics about the path, state `z = [vy, r, e_y, e_psi, delta]`,
input `delta_cmd`, measured disturbance `kappa_ref`.

Derived directly from the plant's own sign conventions: the tyre produces
`Fy = -C_alpha * alpha` with `alpha_f = (vy + a r)/vx - delta` and
`alpha_r = (vy - b r)/vx`, so `Fy_f = Cf (delta - (vy + a r)/vx)`.
`delta` is a state because the plant has a first-order steering actuator;
modelling it here keeps the controller consistent with the plant it drives.
"""
function error_dynamics(v::VehicleLin, vx)
    Ac = [ -(v.Cf+v.Cr)/(v.m*vx)         -(v.a*v.Cf-v.b*v.Cr)/(v.m*vx)-vx          0.0  0.0  v.Cf/v.m;
           -(v.a*v.Cf-v.b*v.Cr)/(v.Iz*vx) -(v.a^2*v.Cf+v.b^2*v.Cr)/(v.Iz*vx)       0.0  0.0  v.a*v.Cf/v.Iz;
            1.0                            0.0                                     0.0  vx   0.0;
            0.0                            1.0                                     0.0  0.0  0.0;
            0.0                            0.0                                     0.0  0.0  -1/v.tau_s ]
    Bc = [0.0, 0.0, 0.0, 0.0, 1/v.tau_s]
    Ec = [0.0, 0.0, 0.0, -vx, 0.0]
    return Ac, Bc, Ec
end

"Exact zero-order-hold discretisation of (Ac, Bc, Ec)."
function discretize(Ac, Bc, Ec, Ts)
    n = size(Ac, 1)
    M = zeros(n+2, n+2)
    M[1:n, 1:n] = Ac
    M[1:n, n+1] = Bc
    M[1:n, n+2] = Ec
    E = exp(M * Ts)
    return E[1:n, 1:n], E[1:n, n+1], E[1:n, n+2]
end

"Condensed QP data. Everything here depends only on vx, so it is built once per run."
struct CondensedQP
    Sx::Matrix{Float64}; Su::Matrix{Float64}; Sd::Matrix{Float64}
    A::Matrix{Float64}            # J = U'A U + b'U + const
    H::Matrix{Float64}            # 2A, the Hessian
    Qbar_Su::Matrix{Float64}      # Su' * Qbar, reused for the linear term
    Sbar_e1::Vector{Float64}
    delta_max::Float64
end

function build_qp(v::VehicleLin, mp::MPCParams, vx)
    Ad, Bd, Ed = discretize(error_dynamics(v, vx)..., mp.Ts)
    n, N = size(Ad, 1), mp.N

    # Powers of Ad are precomputed once: the obvious nested-loop version calls
    # Ad^(k-j) inside the inner loop, which is O(N^2) matrix powers and far too
    # slow now that the QP is rebuilt as the operating point moves.
    Apow = Vector{Matrix{Float64}}(undef, N+1)
    Apow[1] = Matrix{Float64}(I, n, n)
    for k in 1:N; Apow[k+1] = Apow[k] * Ad; end

    Sx = zeros(n*N, n); Su = zeros(n*N, N); Sd = zeros(n*N, N)
    for k in 1:N
        Sx[(k-1)*n+1 : k*n, :] = Apow[k+1]
        for j in 1:k
            Aj = Apow[k-j+1]
            Su[(k-1)*n+1 : k*n, j] = Aj * Bd
            Sd[(k-1)*n+1 : k*n, j] = Aj * Ed
        end
    end

    Q = Diagonal([0.0, mp.q_r, mp.q_ey, mp.q_epsi, 0.0])
    Qbar = kron(Matrix{Float64}(I, N, N), Q)

    Dm = Matrix{Float64}(I, N, N)
    for k in 2:N; Dm[k, k-1] = -1.0; end

    A = Symmetric(Su' * Qbar * Su + mp.R*I + mp.S * (Dm' * Dm)) |> Matrix
    # The delta-rate term is S*||Dm*U - c||^2 with c = u_prev*e1, whose linear
    # part is -2*S*u_prev*(Dm'*e1) = -2*S*u_prev*e1. Note Dm'*e1 is e1, not the
    # first column of Dm.
    Se1 = zeros(N); Se1[1] = mp.S
    return CondensedQP(Sx, Su, Sd, A, 2 .* A, Su' * Qbar, Se1, mp.delta_max)
end

"""
    solve_qp!(U, qp, z0, Dseq, u_prev)

Box-constrained QP by a primal-dual active-set method.

Condensing an MPC problem over a 50-step horizon leaves a badly conditioned
Hessian, and a first-order method (coordinate descent, projected gradient) does
not get close in any reasonable number of sweeps — measured here, 80 sweeps
still left the objective 11 % above the optimum with no constraint active. The
problem is only `N x N` with `N = 50`, so a direct solve is both exact and
cheap; the active set is handled by pinning the worst violator and then
releasing any bound whose gradient points back into the feasible region.
"""
function solve_qp!(U, qp::CondensedQP, z0, Dseq, u_prev)
    b = 2 .* (qp.Qbar_Su * (qp.Sx * z0 + qp.Sd * Dseq)) .- 2 .* (qp.Sbar_e1 .* u_prev)
    H, N, dmax = qp.H, length(U), qp.delta_max

    pinned = fill(NaN, N)                    # NaN == free
    for _ in 1:(8N + 20)
        fr = findall(isnan, pinned); fx = findall(!isnan, pinned)

        if isempty(fr)
            U .= pinned                      # every input is at a bound
        else
            rhs = -b[fr]
            isempty(fx) || (rhs .-= H[fr, fx] * pinned[fx])
            uf = H[fr, fr] \ rhs
            U[fr] .= uf
            isempty(fx) || (U[fx] .= pinned[fx])

            worst, wi, wv = 1e-12, 0, 0.0    # primal: pin the worst violation
            for (k, i) in enumerate(fr)
                v = uf[k] > dmax ? uf[k] - dmax : uf[k] < -dmax ? -dmax - uf[k] : 0.0
                if v > worst; worst, wi, wv = v, i, (uf[k] > dmax ? dmax : -dmax); end
            end
            if wi != 0
                pinned[wi] = wv
                continue
            end
        end

        # Dual: release the single worst wrongly-pinned bound. This must also run
        # when `fr` is empty, or an all-pinned iterate is returned unchecked; and
        # releasing one at a time rather than all of them avoids zig-zagging
        # between active sets.
        isempty(fx) && return U
        g = H * U .+ b
        worstd, wd = 1e-10, 0
        for i in fx
            d = pinned[i] > 0 ? g[i] : -g[i]     # >0 means the bound should release
            if d > worstd; worstd, wd = d, i; end
        end
        wd == 0 && return U
        pinned[wd] = NaN
    end
    clamp!(U, -dmax, dmax)                   # safety net if the loop ran out
    return U
end

# ===========================================================================
# Closed loop
# ===========================================================================

"""
    axle_stiffness(integ, P, twin, veh; alpha_eps)

Secant axle cornering stiffness at the current operating point, blended back to
the linear value at small slip.

A constant-stiffness prediction model is only valid while the tyres are linear.
Measured on this course at 31 m/s, the real secant stiffness falls to 0.50 of
the nominal at the front and 0.25 at the rear — the rear worse because it also
carries the drive force through the friction ellipse. An MPC that believes the
car is four times more responsive than it is will command steering, get a
fraction of the predicted yaw, and correct harder: the loop diverges.

Secant rather than tangent stiffness: the tangent goes to zero at the force
peak, which would tell the controller it has no authority at all and send it
straight to the steering limit. The secant reproduces the force actually being
generated at the current slip angle.

The blend weight `w = a^2/(a^2 + alpha_eps^2)` keeps the estimate well posed
near zero slip, where `Fy/alpha` is a ratio of two small numbers.
"""
function axle_stiffness(integ, P, twin::Bool, veh::VehicleLin;
                        alpha_eps = deg2rad(1.0), floor_frac = 0.2)
    if twin
        af  = 0.5*(integ[P.car.fl.alpha] + integ[P.car.fr.alpha])
        ar  = 0.5*(integ[P.car.rl.alpha] + integ[P.car.rr.alpha])
        Fyf = integ[P.car.fl.tire.Fy] + integ[P.car.fr.tire.Fy]
        Fyr = integ[P.car.rl.tire.Fy] + integ[P.car.rr.tire.Fy]
    else
        af  = integ[P.car.front.alpha];      ar  = integ[P.car.rear.alpha]
        Fyf = integ[P.car.front.tire.Fy];    Fyr = integ[P.car.rear.tire.Fy]
    end
    blend(a, Fy, C0) = begin
        w = a^2 / (a^2 + alpha_eps^2)
        C = (1 - w)*C0 + w*abs(Fy)/max(abs(a), 1e-9)
        clamp(C, floor_frac*C0, C0)
    end
    return blend(af, Fyf, veh.Cf), blend(ar, Fyr, veh.Cr)
end

"Build and simplify a plant once; reused across every run."
function build_plant(which::Symbol)
    ctor = which === :doubletrack ? TestVehicleDynamics.DLCPlantDoubleTrack :
           which === :bicycle     ? TestVehicleDynamics.DLCPlantBicycle :
           error("unknown plant $which")
    return DyadInterface.simplify_model(ctor(; name = which))
end

"""
    run_dlc(P, vx0; path, mpc, veh, tmax)

Simulate the closed loop. The MPC runs at `mpc.Ts`, writing `act.delta_cmd`;
the plant integrates between samples, so this is genuine sampled-data control
with a zero-order hold.
"""
function run_dlc(P, vx0; course = DLCCourse(), path = reference(course),
                 mpc = MPCParams(), veh = VehicleLin(), tmax = nothing,
                 record_corners = false, adaptive = true, relin_tol = 0.02)
    twin = nameof(P) === :doubletrack
    qp = build_qp(veh, mpc, vx0)
    Cf_qp, Cr_qp = veh.Cf, veh.Cr        # stiffnesses the current QP was built with
    nrelin = 0
    s_end = total_length(course)
    T = tmax === nothing ? 1.6 * s_end / vx0 + 1.0 : tmax

    prob = ODEProblem(P, Pair{Any,Any}[P.vx0 => vx0], (0.0, T))
    integ = init(prob, nothing; abstol = 1e-8, reltol = 1e-8, save_everystep = false)

    t = Float64[]; X = Float64[]; Y = Float64[]; psi = Float64[]
    vx = Float64[]; vy = Float64[]; r = Float64[]; dcmd = Float64[]
    ay = Float64[]; beta = Float64[]
    # corner loads and the suspension-lagged accelerations, for the 3D view
    Fz = [Float64[] for _ in 1:4]; ax_lag = Float64[]; ay_lag = Float64[]

    U = zeros(mpc.N); u_prev = 0.0; Dseq = zeros(mpc.N)
    ok = true

    while integ.t < T - 1e-9
        Xn = integ[P.car.X]; Yn = integ[P.car.Y]; pn = integ[P.car.psi]
        vxn = integ[P.car.vx]; vyn = integ[P.car.vy]; rn = integ[P.car.r]
        dn  = integ[P.act.delta_act]

        push!(t, integ.t); push!(X, Xn); push!(Y, Yn); push!(psi, pn)
        push!(vx, vxn); push!(vy, vyn); push!(r, rn); push!(dcmd, u_prev)
        push!(ay, integ[P.car.ay]); push!(beta, integ[P.car.beta])
        if record_corners
            push!(Fz[1], integ[P.car.fl.Fz]); push!(Fz[2], integ[P.car.fr.Fz])
            push!(Fz[3], integ[P.car.rl.Fz]); push!(Fz[4], integ[P.car.rr.Fz])
            push!(ax_lag, integ[P.car.ax_lag]); push!(ay_lag, integ[P.car.ay_lag])
        end

        Xn > s_end + 10 && break                      # cleared the course
        (vxn < 1.0 || abs(Yn) > 25.0) && (ok = false; break)

        # station is taken as X: the path runs along X and heading stays small,
        # so arc length and X agree to well under 2% over the course
        e_y   = Yn - y_ref(path, Xn)
        e_psi = pn - psi_ref(path, Xn)
        z0 = [vyn, rn, e_y, e_psi, dn]

        # curvature preview: the MPC sees the path ahead, which is the whole
        # point of using it here rather than a pure feedback driver
        @inbounds for k in 1:mpc.N
            Dseq[k] = kappa_ref(path, Xn + vxn * k * mpc.Ts)
        end

        # LTV: re-linearise the prediction model at the current operating point.
        # Rebuilt only when the stiffness has drifted, since condensing the
        # horizon is the expensive part of the loop.
        if adaptive
            Cf_now, Cr_now = axle_stiffness(integ, P, twin, veh)
            if abs(Cf_now - Cf_qp) > relin_tol*Cf_qp || abs(Cr_now - Cr_qp) > relin_tol*Cr_qp
                Cf_qp, Cr_qp = Cf_now, Cr_now
                qp = build_qp(VehicleLin(; m = veh.m, Iz = veh.Iz, a = veh.a, b = veh.b,
                                          Cf = Cf_qp, Cr = Cr_qp, tau_s = veh.tau_s),
                              mpc, vxn)
                nrelin += 1
            end
        end

        solve_qp!(U, qp, z0, Dseq, u_prev)
        u_prev = U[1]
        integ.ps[P.act.delta_cmd] = u_prev
        circshift!(U, -1); U[end] = U[end-1]          # shift for the warm start

        try
            step!(integ, mpc.Ts, true)
        catch
            ok = false; break
        end
        integ.sol.retcode in (ReturnCode.Success, ReturnCode.Default) || (ok = false; break)
    end

    return (; t, X, Y, psi, vx, vy, r, delta = dcmd, ay, beta, Fz, ax_lag, ay_lag, ok,
             nrelin, cleared = !isempty(X) && X[end] > s_end)
end

"Pass if the course was cleared, the run stayed valid, and no corner left a gate."
function evaluate(P, vx0; course = DLCCourse(), path = reference(course),
                  mpc = MPCParams(), veh = VehicleLin())
    traj = run_dlc(P, vx0; course, path, mpc, veh)
    margin = (traj.ok && traj.cleared) ? score(traj, gates(course)) : -Inf
    spun = isempty(traj.beta) ? true : maximum(abs, traj.beta) > deg2rad(20)
    return (; traj, margin, pass = traj.ok && traj.cleared && !spun && margin > 0)
end

"""
    max_pass_speed(P; lo, hi, tol)

Bisect entry speed on the pass criterion. `lo` must pass and `hi` must fail;
both are checked, because a silently failing bracket returns a meaningless
number.
"""
function max_pass_speed(P; lo = 12.0, hi = 40.0, tol = 0.1, kwargs...)
    evaluate(P, lo; kwargs...).pass || return (speed = NaN, note = "lower bracket $lo m/s already fails")
    evaluate(P, hi; kwargs...).pass && return (speed = NaN, note = "upper bracket $hi m/s still passes")
    while hi - lo > tol
        mid = 0.5*(lo + hi)
        evaluate(P, mid; kwargs...).pass ? (lo = mid) : (hi = mid)
    end
    return (speed = lo, note = "")
end

"""
    best_inset(P; insets, course, mpc)

The planner inset is part of the driver, not of the vehicle: it trades corridor
margin (room for tracking error and body yaw) against path curvature (and hence
the grip-limited speed). Both effects are monotone and opposed, so there is an
interior optimum, and reporting any single inset would understate the car.
"""
function best_inset(P; insets = 0.08:0.01:0.16, course = DLCCourse(),
                    mpc = MPCParams(), tol = 0.05)
    best = (speed = -Inf, inset = NaN, path = nothing)
    for ya in insets
        pr = reference(course; yaw_allow = ya)
        m = max_pass_speed(P; lo = 14.0, hi = 42.0, tol, course, path = pr, mpc)
        if !isnan(m.speed) && m.speed > best.speed
            best = (speed = m.speed, inset = ya, path = pr)
        end
    end
    return best
end

"Per-sample worst corridor margin, for plotting where the run is tight."
function margin_trace(traj, gs)
    out = fill(NaN, length(traj.t))
    for n in eachindex(traj.t)
        m = Inf
        for (cx, cy) in corners(traj.X[n], traj.Y[n], traj.psi[n]), g in gs
            (g.x0 <= cx <= g.x1) || continue
            m = min(m, g.half - abs(cy - g.yc))
        end
        isfinite(m) && (out[n] = m)
    end
    return out
end

"""
    tune_driver(P; course)

Sweep MPC weights and horizon, each optimised over the planner inset. Used to
establish that the controller is not what limits the reported speed.
"""
function tune_driver(P; course = DLCCourse())
    cfgs = [("baseline",             MPCParams()),
            ("q_ey / 4",             MPCParams(q_ey = 12.0)),
            ("q_ey x 4",             MPCParams(q_ey = 192.0)),
            ("S / 6  (freer steer)", MPCParams(S = 5.0)),
            ("S x 3",                MPCParams(S = 90.0)),
            ("R / 10",               MPCParams(R = 0.05)),
            ("N = 50  (1.0 s)",      MPCParams(N = 50)),
            ("N = 200 (4.0 s)",      MPCParams(N = 200))]
    println("\nDriver sensitivity (each row optimised over planner inset)")
    for (nm, mp) in cfgs
        b = best_inset(P; course, mpc = mp)
        @printf("  %-22s v_max = %6.2f m/s (%3.0f km/h)   inset %.2f m\n",
                nm, b.speed, 3.6*b.speed, b.inset)
    end
end

# ===========================================================================
# Report
# ===========================================================================

function main(; course = DLCCourse(), mpc = MPCParams(), do_tuning = false, plot = true)
    gs = gates(course)
    println("Double lane change, constant-speed protocol, MPC driver")
    @printf("course %.0f m, lateral offset %.2f m, vehicle width %.2f m\n",
            total_length(course), course.D, course.w)
    for (i, g) in enumerate(gs)
        @printf("  gate %d  X=[%5.1f,%5.1f]  centre %.2f  half-width %.3f  (CG margin %.3f m)\n",
                i, g.x0, g.x1, g.yc, g.half, g.half - course.w/2)
    end

    results = Dict{Symbol,Any}()
    for which in (:doubletrack, :bicycle)
        P = build_plant(which)
        b = best_inset(P; course, mpc)
        res = evaluate(P, b.speed; course, path = b.path, mpc)
        results[which] = (; b..., res, P)
        @printf("\n%-13s max pass speed %6.2f m/s (%3.0f km/h) at planner inset %.2f m\n",
                string(which), b.speed, 3.6*b.speed, b.inset)
        @printf("  at that speed: peak ay %.2f m/s2, peak |beta| %.2f deg, peak |delta| %.3f rad, margin %.3f m\n",
                maximum(abs, res.traj.ay), rad2deg(maximum(abs, res.traj.beta)),
                maximum(abs, res.traj.delta), res.margin)
        @printf("  reference kappa_max %.5f (R = %.1f m) -> grip-limited speed sqrt(ay/kappa) = %.1f m/s\n",
                maximum(abs, b.path.kap), 1/maximum(abs, b.path.kap),
                sqrt(maximum(abs, res.traj.ay)/maximum(abs, b.path.kap)))
    end

    do_tuning && tune_driver(results[:doubletrack].P; course)
    plot && plot_dlc(results, course)
    return results
end

# ===========================================================================
# Figure
# ===========================================================================

const OUTFIG = joinpath(@__DIR__, "dlc_mpc.png")

"""
Steering ratio used to report a steering-wheel angle.

The 3-DOF model has no steering column: `delta` is the road-wheel angle. This
is a typical passenger-car rack ratio applied for reporting only — changing it
rescales the plots and the 3D gauge and changes no physics.
"""
const STEER_RATIO = 16.0

function plot_dlc(results, course)
    gs = gates(course)
    D  = results[:doubletrack]; tr = D.res.traj; pth = D.path
    tb = results[:bicycle].res.traj
    mD = margin_trace(tr, gs); mB = margin_trace(tb, gs)
    Plots.gr(size = (1400, 1380))

    gateshape!(p, g, lbl) = Plots.plot!(p,
        [g.x0, g.x1, g.x1, g.x0, g.x0],
        [g.yc-g.half, g.yc-g.half, g.yc+g.half, g.yc+g.half, g.yc-g.half],
        seriestype = :shape, fc = :gray88, lc = :gray55, lw = 1.5, label = lbl)

    # --- full course. Lateral scale is deliberately exaggerated: at true aspect
    #     a 125 x 8 m course is a 15:1 strip in which nothing is legible.
    p1 = Plots.plot(title = @sprintf("Course (lateral scale exaggerated) — double-track limit %.1f m/s (%.0f km/h)",
                                     D.speed, 3.6*D.speed),
                    xlabel = "X [m]", ylabel = "Y [m]", legend = :topleft, ylims = (-1.8, 5.4))
    for (i, g) in enumerate(gs); gateshape!(p1, g, i == 1 ? "cone gates" : ""); end
    ss = 0:0.5:total_length(course)
    Plots.plot!(p1, ss, [y_ref(pth, s) for s in ss], lc = :orangered, ls = :dash, lw = 2,
                label = "min-curvature reference")
    Plots.plot!(p1, tr.X, tr.Y, lc = :navy,     lw = 2.5, label = "CG path, double-track")
    Plots.plot!(p1, tb.X, tb.Y, lc = :seagreen, lw = 2, ls = :dot, label = "CG path, bicycle")

    # --- true-aspect zoom on whichever gate binds
    nb = argmin(replace(mD, NaN => Inf))
    xb = tr.X[nb]
    p2 = Plots.plot(title = @sprintf("True aspect at the binding gate (X = %.0f m)", xb),
                    xlabel = "X [m]", ylabel = "Y [m]", legend = false,
                    aspect_ratio = :equal, xlims = (xb-13, xb+13))
    for g in gs; gateshape!(p2, g, ""); end
    Plots.plot!(p2, tr.X, tr.Y, lc = :navy, lw = 2)
    for n in eachindex(tr.t)
        abs(tr.X[n] - xb) < 12 || continue
        n % 4 == 0 || continue
        o = corners(tr.X[n], tr.Y[n], tr.psi[n])[[1,2,4,3,1]]
        Plots.plot!(p2, first.(o), last.(o), lc = :navy, lw = 1, alpha = 0.5)
    end

    p4 = Plots.plot(title = "Lateral acceleration", xlabel = "X [m]", ylabel = "a_y [m/s²]", legend = :topright)
    Plots.plot!(p4, tr.X, tr.ay, lc = :navy, lw = 2, label = "double-track")
    Plots.plot!(p4, tb.X, tb.ay, lc = :seagreen, lw = 2, ls = :dot, label = "bicycle")
    Plots.hline!(p4, [9.0, -9.0], lc = :red, ls = :dash, lw = 1.5, label = "open-loop peak grip")

    p5 = Plots.plot(title = "Corridor margin, worst corner", xlabel = "X [m]",
                    ylabel = "margin [m]", legend = :topright)
    Plots.plot!(p5, tr.X, mD, lc = :navy, lw = 2, label = "double-track")
    Plots.plot!(p5, tb.X, mB, lc = :seagreen, lw = 2, ls = :dot, label = "bicycle")
    Plots.hline!(p5, [0.0], lc = :red, ls = :dash, lw = 1.5, label = "cone contact")

    # Time histories. Everything above is plotted against station X, which hides
    # the fact that the two cars run the course at different speeds; these two
    # are against time, which is what a driver experiences.
    p6 = Plots.plot(title = "Steering wheel angle ($(Int(STEER_RATIO)):1 ratio)",
                    xlabel = "t [s]", ylabel = "SW angle [deg]", legend = :topright)
    Plots.plot!(p6, tr.t, rad2deg.(tr.delta .* STEER_RATIO), lc = :navy, lw = 2, label = "double-track")
    Plots.plot!(p6, tb.t, rad2deg.(tb.delta .* STEER_RATIO), lc = :seagreen, lw = 2, ls = :dot, label = "bicycle")
    Plots.hline!(p6, [0.0], lc = :gray60, lw = 1, label = "")

    p7 = Plots.plot(title = "Yaw angle", xlabel = "t [s]", ylabel = "ψ [deg]", legend = :topright)
    Plots.plot!(p7, tr.t, rad2deg.(tr.psi), lc = :navy, lw = 2, label = "double-track")
    Plots.plot!(p7, tb.t, rad2deg.(tb.psi), lc = :seagreen, lw = 2, ls = :dot, label = "bicycle")
    Plots.hline!(p7, [0.0], lc = :gray60, lw = 1, label = "")

    # The true-aspect panel gets a full-width row of its own: it is a wide, short
    # box by construction, and sharing a row with a normal panel wastes most of
    # the height to whitespace.
    Plots.plot(p1, p2, p4, p5, p6, p7,
               layout = Plots.@layout([a{0.19h}; b{0.15h}; c d; e f]),
               left_margin = 10Measures.mm, bottom_margin = 7Measures.mm,
               top_margin = 4Measures.mm, right_margin = 4Measures.mm,
               titlefontsize = 10)
    Plots.savefig(OUTFIG)
    @info "figure written" OUTFIG
    return OUTFIG
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; do_tuning = true)
end

