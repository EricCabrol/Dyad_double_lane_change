#
# presentation.jl
#
# Builds a self-contained HTML deck summarising the whole vehicle-dynamics study.
#
# Run from the package root:
#     julia --project=. scripts/presentation.jl
#
# Output: scripts/presentation.html — one file, no external dependencies, images
# embedded as data URIs. The interactive 3D scenes are linked relatively, so keep
# the deck in scripts/ alongside them.
#

include(joinpath(@__DIR__, "visualize_dlc_3d.jl"))   # pulls in dlc_mpc.jl + geometry

using Base64
using Plots, Measures

const DECK     = joinpath(@__DIR__, "presentation.html")
const FIG_TYRE = joinpath(@__DIR__, "fig_tyre.png")
const FIG_SCENE = joinpath(@__DIR__, "fig_scene.png")

# ===========================================================================
# Figure: tyre characteristic
# ===========================================================================

function tyre_figure()
    sysT  = TestVehicleDynamics.TestTireSweep(; name = :ts)
    ssysT = DyadInterface.simplify_model(sysT)
    sweep(; amax, Fz, Fx) = solve(
        ODEProblem(ssysT, Pair{Any,Any}[ssysT.alpha_max => amax, ssysT.T_sweep => 1.0,
                                        ssysT.Fz_test => Fz, ssysT.Fx_test => Fx],
                   (0.0, 1.0)); abstol = 1e-10, reltol = 1e-10,
        saveat = range(0, 1.0, length = 400))

    # Plotted as cornering force (-F_y): the model's F_y is negative for positive
    # slip because it is restoring, but the curve reads better the usual way up.
    Plots.gr(size = (1250, 430))
    p1 = Plots.plot(title = "Pure slip: load sensitivity", xlabel = "slip angle α [deg]",
                    ylabel = "cornering force −F_y [N]", legend = :bottomright)
    for (Fz, c) in zip((2000.0, 4000.0, 6000.0, 8000.0), (:steelblue, :navy, :darkorange, :firebrick))
        s = sweep(amax = 0.35, Fz = Fz, Fx = 0.0)
        Plots.plot!(p1, rad2deg.(s[ssysT.alpha_src.y]), -s[ssysT.tire.Fy], lw = 2.5, lc = c,
                    label = "F_z = $(Int(Fz)) N   (μ = $(round(s[ssysT.tire.mu][1], digits=3)))")
    end
    # Explanation lives in the slide text, not on the axes: there is no empty
    # region large enough here that does not collide with a curve or the legend.

    p2 = Plots.plot(title = "Combined slip: friction ellipse (F_z = 4000 N)",
                    xlabel = "slip angle α [deg]", ylabel = "cornering force −F_y [N]",
                    legend = :bottomright)
    for (f, c) in zip((0.0, 0.5, 0.8, 1.0), (:navy, :seagreen, :darkorange, :firebrick))
        s = sweep(amax = 0.35, Fz = 4000.0, Fx = f*4000.0)
        Plots.plot!(p2, rad2deg.(s[ssysT.alpha_src.y]), -s[ssysT.tire.Fy], lw = 2.5, lc = c,
                    label = "F_x/μF_z = $(f)   →  g = $(round(s[ssysT.tire.g_ell][1], digits=3))")
    end


    Plots.plot(p1, p2, layout = (1, 2), left_margin = 9Measures.mm,
               bottom_margin = 7Measures.mm, top_margin = 3Measures.mm, titlefontsize = 10)
    Plots.savefig(FIG_TYRE)
    return FIG_TYRE
end

# ===========================================================================
# Figure: 3D scene still
#
# PlotlyKaleido cannot start in this runtime (its binary artifact points at a
# CI path that does not exist locally), so the still is rendered here from the
# same meshes the interactive scene uses: orthographic projection plus a
# painter's-algorithm depth sort.
# ===========================================================================

"Orthonormal screen basis for a view direction pointing from camera into scene."
function view_basis(dir)
    d = normalize(dir)
    right = normalize(cross(d, [0.0, 0.0, 1.0]))
    up = cross(right, d)
    return d, right, up
end

"Collect (depth, xs, ys, colour) for every triangle of every mesh."
function collect_faces(meshes_colors, d, right, up)
    faces = Tuple{Float64,Vector{Float64},Vector{Float64},Any}[]
    for (mh, col) in meshes_colors
        X, Y, Z, I, J, K = mh
        for q in eachindex(I)
            idx = (I[q]+1, J[q]+1, K[q]+1)
            P = [[X[t], Y[t], Z[t]] for t in idx]
            push!(faces, (sum(dot(p, d) for p in P)/3,
                          [dot(p, right) for p in P], [dot(p, up) for p in P], col))
        end
    end
    return faces
end

function scene_still(; vx0 = nothing)
    course = DLCCourse(); P = build_plant(:doubletrack)
    best = vx0 === nothing ? best_inset(P; course, mpc = MPCParams()) :
                             (speed = vx0, inset = 0.11, path = reference(course; yaw_allow = 0.11))
    M = sample_dlc(; vx0 = best.speed, course, path = best.path)
    cones = cone_positions(course); hit = struck(cones, M.full)

    # Choose the most-yawed frame that is still inside a coned section: it has both
    # cones to give the corridor scale and visible body attitude. The peak steering
    # of the whole run happens in a transition, where there is nothing to show.
    gs = gates(course)
    cand = [q for q in eachindex(M.X) if any(g -> g.x0 <= M.X[q] <= g.x1, gs)]
    n = cand[argmax(abs.(M.psi[cand]))]

    lo, hi = extrema(reduce(vcat, M.Fz))
    d, right, up = view_basis([0.95, 1.0, -0.78])

    mc = Tuple{Any,Any}[]
    cx, cy = M.X[n], M.Y[n]
    push!(mc, (box_mesh([cx - 1.0, cy + 1.0, -0.04], Matrix{Float64}(I(3)), (13.0, 9.0, 0.02)),
               RGB(0.93,0.93,0.92)))
    for (ci, (px, py)) in enumerate(cones)
        (abs(px - cx) < 13 && abs(py - cy) < 9) || continue
        push!(mc, (cone_mesh([px, py, 0.0], CONE_R, CONE_H), hit[ci] ? RGB(0.82,0.10,0.10) : RGB(0.94,0.51,0.12)))
    end
    for (mh, col) in dynamic_parts(M, n, lo, hi)
        m = match(r"rgb\((\d+),(\d+),(\d+)\)", col)
        push!(mc, (mh, RGB(parse(Int,m[1])/255, parse(Int,m[2])/255, parse(Int,m[3])/255)))
    end

    faces = collect_faces(mc, d, right, up)
    sort!(faces, by = f -> -f[1])                 # far first

    Plots.gr(size = (1250, 620))
    p = Plots.plot(aspect_ratio = :equal, legend = false, grid = false, axis = false,
                   ticks = false, framestyle = :none,
                   title = @sprintf("Double lane change at the limit — %.1f m/s (%.0f km/h), steering wheel %+.0f°",
                                    best.speed, 3.6*best.speed, rad2deg(M.delta[n]*STEER_RATIO)),
                   titlefontsize = 11)
    for (_, xs, ys, col) in faces
        Plots.plot!(p, xs, ys, seriestype = :shape, fc = col, lc = col, lw = 0.2)
    end
    # driven path on top
    sel = [q for q in eachindex(M.full.X) if abs(M.full.X[q] - cx) < 13]
    Plots.plot!(p, [dot([M.full.X[q], M.full.Y[q], 0.07], right) for q in sel],
                   [dot([M.full.X[q], M.full.Y[q], 0.07], up) for q in sel],
                lc = RGB(0.15,0.25,0.55), lw = 2.5)
    Plots.savefig(FIG_SCENE)
    return FIG_SCENE
end

# ===========================================================================
# Deck
# ===========================================================================

img(path) = "data:image/png;base64," * base64encode(read(path))

const CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#eceff3;color:#1b2430;overflow:hidden}
#deck{height:100vh;width:100vw;position:relative}
.slide{position:absolute;inset:0;padding:38px 58px 64px;background:#fff;
       display:none;flex-direction:column;overflow-y:auto}
.slide.on{display:flex}
h1{font-size:40px;font-weight:650;letter-spacing:-.5px;margin-bottom:10px}
h2{font-size:27px;font-weight:620;letter-spacing:-.3px;margin-bottom:14px;color:#102a43;
   border-bottom:2px solid #d7dee8;padding-bottom:8px}
h3{font-size:17px;font-weight:620;margin:14px 0 6px;color:#243b53}
p,li{font-size:15.5px;line-height:1.55;color:#33475b}
ul{margin:6px 0 6px 20px}li{margin:4px 0}
.sub{font-size:19px;color:#5a6b7f;font-weight:400}
.cols{display:flex;gap:26px;align-items:flex-start}.cols>*{flex:1;min-width:0}
code{font-family:'SF Mono',Consolas,monospace;font-size:13px;background:#eef2f7;
     padding:1px 5px;border-radius:3px;color:#0b4f6c}
pre{background:#1b2430;color:#e3e9f0;padding:13px 16px;border-radius:6px;
    font-family:'SF Mono',Consolas,monospace;font-size:12.5px;line-height:1.5;
    overflow-x:auto;margin:8px 0}
pre .k{color:#7fd3ff}pre .c{color:#7c8b9c;font-style:italic}pre .s{color:#a5e6a5}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #dde4ec}
th{background:#f4f7fa;font-weight:620;color:#102a43}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.good{color:#1b7a3d;font-weight:600}.bad{color:#b4232a;font-weight:600}
figure{margin:8px 0;text-align:center}
figure img{max-width:100%;max-height:60vh;border:1px solid #dde4ec;border-radius:5px}
figcaption{font-size:13px;color:#6b7d91;margin-top:6px}
.note{background:#fff8e6;border-left:4px solid #e0a800;padding:11px 15px;
      border-radius:0 5px 5px 0;margin:10px 0;font-size:14.5px}
.key{background:#eaf4ff;border-left:4px solid #2176ae;padding:11px 15px;
     border-radius:0 5px 5px 0;margin:10px 0;font-size:14.5px}
.warn{background:#fdeeee;border-left:4px solid #b4232a;padding:11px 15px;
      border-radius:0 5px 5px 0;margin:10px 0;font-size:14.5px}
.cards{display:flex;gap:16px;margin:14px 0;flex-wrap:wrap}
.card{flex:1;min-width:210px;border:1px solid #d7dee8;border-radius:7px;padding:14px 16px;
      background:#fafbfd;text-decoration:none;color:inherit;display:block;transition:.14s}
.card:hover{border-color:#2176ae;background:#f2f8ff;transform:translateY(-2px)}
.card .t{font-weight:620;font-size:15px;color:#102a43;margin-bottom:4px}
.card .d{font-size:13px;color:#6b7d91;line-height:1.45}
.files{font-family:'SF Mono',Consolas,monospace;font-size:13px;line-height:1.75}
.files .d{color:#2176ae;font-weight:600}.files .m{color:#8595a7}
#nav{position:absolute;bottom:14px;right:26px;font-size:13px;color:#8595a7;
     display:flex;gap:14px;align-items:center;z-index:10}
#nav button{border:1px solid #d7dee8;background:#fff;border-radius:5px;
            padding:4px 11px;cursor:pointer;font-size:14px;color:#33475b}
#nav button:hover{border-color:#2176ae;color:#2176ae}
#bar{position:absolute;bottom:0;left:0;height:3px;background:#2176ae;transition:width .2s}
.tag{display:inline-block;background:#e3ecf5;color:#2176ae;border-radius:3px;
     padding:1px 7px;font-size:12px;font-weight:600;margin-right:6px}
"""

const JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
function go(n){i=Math.max(0,Math.min(S.length-1,n));
  S.forEach((s,k)=>s.classList.toggle('on',k===i));
  document.getElementById('num').textContent=(i+1)+' / '+S.length;
  document.getElementById('bar').style.width=((i+1)/S.length*100)+'%';
  S[i].scrollTop=0;}
document.addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' '].includes(e.key)){go(i+1);e.preventDefault();}
  if(['ArrowLeft','PageUp'].includes(e.key)){go(i-1);e.preventDefault();}
  if(e.key==='Home')go(0); if(e.key==='End')go(S.length-1);});
go(0);
"""

slide(body) = "<section class=\"slide\">$body</section>\n"

function build_deck(results)
    D = results[:doubletrack]; B = results[:bicycle]
    s = IOBuffer()

    # 1 title
    write(s, slide("""
      <div style="margin:auto 0">
      <h1>A 3-DOF Vehicle Dynamics Study in Dyad</h1>
      <p class="sub" style="margin-bottom:26px">From a Magic Formula tyre to a closed-loop MPC double lane change</p>
      <div class="cols">
        <div><h3>Models</h3><ul>
          <li>Pacejka tyre with load sensitivity and a friction ellipse</li>
          <li>Single-track and double-track 3-DOF vehicles sharing one base</li>
          <li>Closed-loop plant with speed hold and a steering actuator</li></ul></div>
        <div><h3>Studies</h3><ul>
          <li>Step steer: where a bicycle model stops being enough</li>
          <li>Double lane change with an MPC driver</li>
          <li>Maximum speed through the gates, and what limits it</li></ul></div>
        <div><h3>Visualisation</h3><ul>
          <li>Interactive 3D scenes with cone gates</li>
          <li>Wheel colour encoding vertical load</li>
          <li>Live steering-wheel gauge and time histories</li></ul></div>
      </div></div>"""))

    # 2 what was built
    write(s, slide("""
      <h2>What was built</h2>
      <div class="cols">
      <div><h3>Dyad — the physics</h3><div class="files">
        <span class="d">dyad/</span><br>
        &nbsp;&nbsp;tire.dyad <span class="m">— MagicFormulaTire + sweep harness</span><br>
        &nbsp;&nbsp;wheel_station.dyad <span class="m">— hub kinematics → tyre → body forces</span><br>
        &nbsp;&nbsp;vehicle.dyad <span class="m">— PlanarVehicleBase (partial),</span><br>
        &nbsp;&nbsp;<span class="m">&nbsp;&nbsp;VehicleBicycle3DOF, VehicleDoubleTrack3DOF</span><br>
        &nbsp;&nbsp;maneuvers.dyad <span class="m">— step-steer harnesses + analyses</span><br>
        &nbsp;&nbsp;plant.dyad <span class="m">— DriverActuator + closed-loop plants</span>
      </div></div>
      <div><h3>Julia — algorithms and views</h3><div class="files">
        <span class="d">scripts/</span><br>
        &nbsp;&nbsp;compare_vehicle_models.jl <span class="m">— step-steer study</span><br>
        &nbsp;&nbsp;dlc_mpc.jl <span class="m">— DLC course, MPC, max-speed search</span><br>
        &nbsp;&nbsp;vehicle_3d_geometry.jl <span class="m">— shared mesh primitives</span><br>
        &nbsp;&nbsp;visualize_vehicle_3d.jl <span class="m">— step-steer 3D scenes</span><br>
        &nbsp;&nbsp;visualize_dlc_3d.jl <span class="m">— DLC 3D scenes</span><br>
        &nbsp;&nbsp;presentation.jl <span class="m">— this deck</span>
      </div></div></div>
      <div class="key"><b>The split is deliberate.</b> Physics lives in Dyad, where the
      acausal connections and unit checking help. Algorithms that are not equations —
      a QP solver, a path planner, a bisection search — live in Julia. The plant never
      needs the reference path; only the controller and the scorer do.</div>"""))

    # 3 tyre
    write(s, slide("""
      <h2>The tyre sets everything else</h2>
      <div class="cols">
      <div><p>Magic Formula lateral force with <b>load-sensitive friction</b> and a
      friction-ellipse derating for combined slip:</p>
      <pre><span class="c"># dyad/tire.dyad</span>
F_z_pos = 0.5*(F_z + sqrt(F_z^2 + F_z_eps^2))
mu      = mu0*(1 - k_mu*(F_z_pos/F_z_nom - 1))
D       = mu*F_z_pos
phi     = B*alpha - E*(B*alpha - atan(B*alpha))
F_y_pure = -D*sin(C*atan(phi))
e_x     = F_x/D
g_ell   = sqrt(0.5*((1-e_x^2) + sqrt((1-e_x^2)^2 + eps^2)))
F_y     = g_ell*F_y_pure</pre>
      <p>Every formulation choice here is about the solver seeing something smooth:
      a rectified <code>F_z</code> so wheel lift cannot go negative, and a smoothed
      square root so the ellipse collapses gracefully instead of going imaginary.</p></div>
      <div><h3>Validated against known answers</h3>
      <table><tr><th>Check</th><th>Expected</th><th>Measured</th></tr>
      <tr><td>peak |F_y|</td><td class="num">μ·F_z = 4000</td><td class="num good">4000.00006</td></tr>
      <tr><td>cornering stiffness</td><td class="num">B·C·D = 76000</td><td class="num good">76000</td></tr>
      <tr><td>peak slip angle</td><td class="num">8–12°</td><td class="num good">10.3°</td></tr>
      <tr><td>ellipse at F_x/D = 0.6</td><td class="num">0.800</td><td class="num good">0.800</td></tr>
      <tr><td>ellipse at F_x/D = 1.0</td><td class="num">0</td><td class="num good">0.007</td></tr>
      </table>
      <div class="note"><b>k_mu is not decoration.</b> With <code>D = μ·F_z</code> strictly
      linear in load, two tyres at F_z/2 are exactly one tyre at F_z — a double-track model
      would then be algebraically identical to a bicycle. Load sensitivity is what makes
      lateral load transfer cost grip at all.</div></div></div>"""))

    # 4 tyre figure
    write(s, slide("""
      <h2>Tyre characteristic</h2>
      <figure><img src="$(img(FIG_TYRE))"><figcaption>
      Left: peak force is μ(F_z)·F_z and μ falls with load, so the curves are degressive —
      doubling the load does not double the grip. Right: the friction ellipse derates
      lateral capacity by √(1−e_x²), matched to four digits.</figcaption></figure>"""))

    # 5 architecture
    write(s, slide("""
      <h2>Two vehicles, one base</h2>
      <div class="cols"><div>
      <pre><span class="k">partial component</span> PlanarVehicleBase
  <span class="c">// the three degrees of freedom</span>
  m*ax = Fx_tires - F_aero - F_roll
  ax   = der(vx) - vy*r
  m*ay = Fy_tires
  ay   = der(vy) + vx*r
  Izz*der(r) = Mz_tires
  <span class="c">// trajectory, for plotting</span>
  der(X) = vx*cos(psi) - vy*sin(psi)
  der(Y) = vx*sin(psi) + vy*cos(psi)
  der(psi) = r</pre>
      <p><code>VehicleBicycle3DOF</code> adds two <code>WheelStation</code>s,
      <code>VehicleDoubleTrack3DOF</code> adds four. Both use the <i>same</i> station
      component, so corner physics is identical and the only structural difference is
      lateral load transfer and track-width yaw moment.</p></div>
      <div><h3>States after simplification</h3>
      <table><tr><th></th><th>bicycle</th><th>double-track</th></tr>
      <tr><td>v<sub>x</sub>, v<sub>y</sub>, r</td><td class="num">3</td><td class="num">3</td></tr>
      <tr><td>X, Y, ψ (quadrature)</td><td class="num">3</td><td class="num">3</td></tr>
      <tr><td>load-transfer lags</td><td class="num">1</td><td class="num">2</td></tr>
      <tr><th>total</th><th class="num">7</th><th class="num">8</th></tr></table>
      <div class="key"><b>Why a lag on load transfer?</b> Quasi-static transfer makes
      F<sub>z</sub> depend on a<sub>y</sub>, which depends on F<sub>y</sub>, which depends
      on F<sub>z</sub> — a nonlinear algebraic loop. A first-order suspension lag
      (τ ≈ 0.05 s) breaks it <i>and</i> is the physical reason roll is not instantaneous.
      The result is a pure ODE: no algebraic states.</div></div></div>"""))

    # 6 validation
    write(s, slide("""
      <h2>Validation before conclusions</h2>
      <p>Every result below was checked against something computed independently, not
      against "the solver returned Success".</p>
      <table>
      <tr><th>Test</th><th>Independent reference</th><th>Agreement</th></tr>
      <tr><td>Straight-line trim</td><td>drag + rolling resistance = 456.3 N</td>
          <td class="good">456.09 N; speed holds to 6 mm/s over 6 s</td></tr>
      <tr><td>Vertical load sum</td><td>m·g = 15696 N</td><td class="good">exact, all four corners</td></tr>
      <tr><td>Yaw-rate gain, both models</td>
          <td>linear bicycle r/δ = v<sub>x</sub>/(L + K<sub>us</sub>v<sub>x</sub>²)</td>
          <td class="good">&lt; 0.16 % error</td></tr>
      <tr><td>Bicycle vs double-track, linear range</td><td>should be identical by construction</td>
          <td class="good">0.05 %</td></tr>
      <tr><td>Steering actuator</td><td>1 − e<sup>−t/τ</sup></td><td class="good">0.9643 vs 0.9643</td></tr>
      <tr><td>MPC prediction model</td><td>nonlinear plant, open loop</td>
          <td class="good">r within 2.4 %, v<sub>y</sub> within 6.4 %</td></tr>
      <tr><td>QP solver</td><td>analytic optimum and KKT conditions</td>
          <td class="good">machine precision, 0–50 active constraints</td></tr>
      </table>
      <div class="key">The linear-range agreement between the two vehicles is the one worth
      dwelling on. It is not a coincidence — it is a prediction. With a load-proportional
      tyre and negligible roll transfer, the double-track <i>must</i> reduce to the bicycle.
      Getting 0.05 % is evidence both models are wired correctly.</div>"""))

    # 7 step steer figure
    write(s, slide("""
      <h2>Step steer: where the bicycle stops being enough</h2>
      <figure><img src="$(img(joinpath(@__DIR__,"compare_vehicle_models.png")))"><figcaption>
      Identical below ~1.5 m/s² — as predicted. They fan out at the limit, and at
      δ = 4.58° the bicycle leaves the stable branch entirely.</figcaption></figure>"""))

    # 8 the physics
    write(s, slide("""
      <h2>Why they diverge</h2>
      <div class="cols"><div>
      <h3>1. Axle capacity loss</h3>
      <p>Transfer ±Δ across an axle carrying F, on a degressive tyre:</p>
      <pre>D₁ + D₂ = μ₀·F·(1 − 4·k_mu·Δ²/F²)</pre>
      <p>The loss is <b>quadratic</b> in transfer and strictly negative. No retuning of a
      bicycle model reproduces it, because a bicycle has no Δ.</p>
      <h3>2. Limit balance</h3>
      <p><code>eps_f</code>, the front share of roll moment, decides which axle saturates
      first. Sweeping it reproduces the classic rule exactly:</p>
      <table><tr><th>eps_f</th><th>α<sub>f</sub>−α<sub>r</sub></th><th>outcome</th></tr>
      <tr><td class="num">0.45</td><td class="num">+28.3</td><td class="bad">spins</td></tr>
      <tr><td class="num">0.50</td><td class="num">−1.42</td><td>stable</td></tr>
      <tr><td class="num">0.55</td><td class="num">−2.61</td><td>stable</td></tr>
      <tr><td class="num">0.65</td><td class="num">−2.96</td><td class="good">most understeer</td></tr>
      </table></div>
      <div><h3>Steady state at 25 m/s</h3>
      <table><tr><th>δ [deg]</th><th>a<sub>y</sub> bic</th><th>a<sub>y</sub> DT</th><th>β bic</th><th>β DT</th></tr>
      <tr><td class="num">0.29</td><td class="num">0.886</td><td class="num">0.886</td><td class="num">−0.16</td><td class="num">−0.16</td></tr>
      <tr><td class="num">1.72</td><td class="num">4.984</td><td class="num">4.894</td><td class="num">−1.01</td><td class="num">−1.01</td></tr>
      <tr><td class="num">2.86</td><td class="num">7.374</td><td class="num">7.018</td><td class="num">−1.86</td><td class="num">−1.82</td></tr>
      <tr><td class="num">3.44</td><td class="num">8.138</td><td class="num">7.651</td><td class="num">−2.29</td><td class="num">−2.20</td></tr>
      <tr><td class="num">4.58</td><td class="num bad">spun</td><td class="num">8.314</td><td class="num bad">−28.0</td><td class="num">−2.75</td></tr>
      <tr><td class="num">5.73</td><td class="num bad">spun</td><td class="num">8.643</td><td class="num bad">−60.0</td><td class="num">−3.20</td></tr>
      </table>
      <div class="note"><b>The bicycle is not merely optimistic, it is qualitatively wrong
      here.</b> It gives up ~6 % more a<sub>y</sub> than it should at 0.8 g, and then goes
      unstable where the real car is comfortable — because with no lateral transfer its
      limit margin between axles is only 2 %, so a step input tips it into oversteer.</div>
      </div></div>"""))

    # 9 validity
    write(s, slide("""
      <h2>Knowing where a model stops being valid</h2>
      <p>The slip angle is regularised so it stays smooth and bounded at standstill:</p>
      <pre>alpha = atan(vy_t / sqrt(vx_t^2 + v_eps^2))</pre>
      <p>The denominator is strictly positive, which is what keeps the model well-behaved at
      low speed — but it also means the tyre always believes it is rolling forwards.</p>
      <div class="warn"><b>This was found, not assumed.</b> An <code>eps_f = 0.35</code> run
      spun, reversed, and reported a<sub>y</sub> = 1.52 m/s² at a sampled time — a
      perfectly plausible-looking number produced entirely outside the model's validity
      envelope. The vehicle was travelling at −8.4 m/s.</div>
      <pre><span class="c">// dyad/vehicle.dyad — PlanarVehicleBase</span>
<span class="s">"Validity envelope: the regularised slip-angle definition assumes
 forward motion, so results are meaningless in reverse"</span>
<span class="k">assert</span> vx > -v_eps</pre>
      <p>The solve now halts at the boundary and returns <code>Unstable</code> rather than
      producing confident nonsense. A model that cannot say when it is out of range is
      more dangerous than one that is simply coarse.</p>"""))

    # 10 3D step steer
    write(s, slide("""
      <h2>Seeing it in 3D</h2>
      <p>Rendering uses <b>PlotlyBase</b>, which ships with the Dyad runtime — nothing to
      install. Left-drag orbits, scroll zooms.</p>
      <div class="cards">
        <a class="card" href="vehicle_3d_follow.html"><div class="t">Step steer — follow camera</div>
          <div class="d">Animated, ±16 m tracking window, play/pause and a time slider.
          Wheel colour is vertical load.</div></a>
        <a class="card" href="vehicle_3d_overview.html"><div class="t">Step steer — overview</div>
          <div class="d">Whole manoeuvre at once with ghost cars and the four tyre
          contact tracks.</div></a>
      </div>
      <div class="cols"><div><h3>Simulated</h3><ul>
        <li>CG path, heading, steer angle</li>
        <li>Per-corner vertical load → wheel colour</li>
        <li>Tyre contact tracks, which separate yaw from translation</li></ul></div>
      <div><h3>Display only</h3><ul>
        <li>Body roll and pitch. The model has <b>no roll or pitch DOF</b>; the attitude
        is reconstructed from <code>ax_lag</code> and <code>ay_lag</code>, the same lagged
        accelerations that drive load transfer. Set the gains to zero and it disappears.</li></ul></div></div>
      <div class="note">Geometry was verified before shipping, via orthographic projections:
      1.2/1.4 m wheelbase split, 1.55 m track, wheels touching the road at r = 0.32 m, body
      leaning onto the outer wheels while the unsprung mass stays flat.</div>"""))

    # 11 closed loop architecture
    write(s, slide("""
      <h2>Closing the loop: a double lane change</h2>
      <div class="cols"><div>
      <h3>Dyad holds the plant</h3>
      <pre><span class="k">component</span> DLCPlantDoubleTrack
  car = VehicleDoubleTrack3DOF(vx0 = vx0)
  act = DriverActuator(vx_target = vx0, ...)
<span class="k">relations</span>
  act.vx_meas = car.vx
  car.Fx_req  = act.Fx
  car.delta   = act.delta</pre>
      <p><code>act.delta_cmd</code> is a free parameter with <b>no binding</b>, so the
      controller writes it at every sample instant; the first-order steering actuator then
      supplies the hold dynamics a real steering system would.</p></div>
      <div><h3>Julia holds the controller</h3>
      <p>A QP solver cannot be written as Dyad relations, and the plant does not need the
      reference path — only the controller and the pass/fail scorer do.</p>
      <pre><span class="c"># genuine sampled-data control, ZOH at 50 Hz</span>
<span class="k">while</span> integ.t &lt; T
    z0 = [vy, r, e_y, e_psi, delta]
    Dseq .= kappa_ref.(path, X .+ vx.*k.*Ts)
    solve_qp!(U, qp, z0, Dseq, u_prev)
    integ.ps[P.act.delta_cmd] = U[1]
    step!(integ, Ts, <span class="k">true</span>)
<span class="k">end</span></pre>
      <div class="key"><b>Protocol: constant speed.</b> A speed-hold PI fights cornering drag,
      so the manoeuvre is a pure lateral-grip limit with no longitudinal load transfer
      helping the front axle. Harsher than the ISO coast-down protocol.</div></div></div>"""))

    # 12 reference path
    write(s, slide("""
      <h2>The mistake that mattered most</h2>
      <p>My first reference was a smoothstep centreline that snaps from lane to lane.
      It is curvature-continuous and looks entirely reasonable — and it is <b>infeasible</b>.</p>
      <table><tr><th>Reference</th><th>min radius</th><th>a<sub>y</sub> at 25 m/s</th><th>verdict</th></tr>
      <tr><td>smoothstep centreline</td><td class="num">44.5 m</td><td class="num">14.0 m/s²</td>
          <td class="bad">beyond any tyre</td></tr>
      <tr><td>minimum-curvature through the corridors</td><td class="num">115 m</td>
          <td class="num">5.4 m/s²</td><td class="good">feasible</td></tr></table>
      <div class="key"><b>Why:</b> the gates are wider than the car, so a real driver cuts
      corners and uses the whole corridor. A centreline throws that away. The difference is
      a factor <b>2.6</b> in required lateral acceleration — about 1.6× in achievable speed.
      Tracking the wrong reference would have measured the planner, not the car.</div>
      <p>The fix is a planner, not a formula: minimise the discrete ∑(y″)² subject to
      corridor boxes. That is a box-constrained QP, solved exactly by an active-set loop —
      solve equality-constrained, pin the worst violator, repeat.</p>
      <div class="note"><b>ISO 3888-2 is infeasible for this car</b> and the code says so
      rather than returning a kinked path. At moose-test widths a 1.76 m body leaves the CG
      a <i>negative</i> margin in gate 3. The symptom when I first ignored it was
      κ = 0.59 (R = 1.7 m); <code>reference()</code> now raises an error instead.</div>"""))

    # 13 MPC
    write(s, slide("""
      <h2>The MPC driver</h2>
      <div class="cols"><div>
      <p>Prediction on path-relative error dynamics, derived directly from the plant's own
      tyre sign conventions, with δ as a state so the controller models the actuator it drives:</p>
      <pre>z = [vy, r, e_y, e_psi, delta]
u = delta_cmd       d = kappa_ref</pre>
      <ul>
        <li>Exact ZOH discretisation, T<sub>s</sub> = 20 ms</li>
        <li>Condensed over N = 100 (2.0 s = 65 m of preview at the limit)</li>
        <li><b>Curvature preview</b> — the controller sees the path ahead, which is the
        whole reason to use MPC here rather than pure feedback</li>
        <li>Input box constraint |δ| ≤ δ<sub>max</sub>; δ-rate penalised</li>
        <li>Prediction model <b>re-linearised at the operating point</b> (LTV)</li>
      </ul>
      <h3>Is the horizon long enough?</h3>
      <p>Judged on convergence of the applied first move <code>U[1]</code> — the only part
      of the solution that reaches the plant:</p>
      <table><tr><th>N</th><th>horizon</th><th>U[1] error</th><th>v<sub>max</sub></th></tr>
      <tr><td class="num">25</td><td>0.5 s</td><td class="bad">up to 250 %</td><td class="num">32.07</td></tr>
      <tr><td class="num">50</td><td>1.0 s</td><td class="bad">up to 5.5 %</td><td class="num">32.51</td></tr>
      <tr><td class="num">75</td><td>1.5 s</td><td class="num">0.5 %</td><td class="num">32.73</td></tr>
      <tr><td class="num"><b>100</b></td><td><b>2.0 s</b></td><td class="good">converged</td><td class="num"><b>32.79</b></td></tr>
      <tr><td class="num">150</td><td>3.0 s</td><td class="good">converged</td><td class="num">32.81</td></tr></table>
      <div class="note">The horizon requirement is <b>coupled to the cost scaling</b>.
      With the old aggressive <code>R = 0.5</code> the closed loop was fast enough that
      even 1.0 s looked converged; softening <code>R</code> slowed it and lengthened the
      preview it needs. Tuning weights and horizon independently would have left 0.9 % of
      the speed on the table.</div></div>
      <div><h3>The solver had to be replaced</h3>
      <p>Condensing over 50 steps leaves cond(H) ≈ 5×10³. Projected coordinate descent,
      the obvious cheap choice, <b>does not converge</b>:</p>
      <table><tr><th>method</th><th>objective</th><th>vs optimum</th></tr>
      <tr><td>coordinate descent, 80 sweeps</td><td class="num">−26.21</td><td class="bad">+11 %</td></tr>
      <tr><td>active set (direct solve)</td><td class="num">−29.3876</td><td class="good">exact</td></tr></table>
      <p>With no constraint even active. The problem is only 50×50, so a direct solve is
      both exact and cheap; the active set pins the worst violator, then releases any bound
      whose gradient points back inside.</p>
      <div class="note">KKT residuals at machine precision across 0, 27, 44, 48 and 50
      active constraints — including the all-pinned case, which an earlier version returned
      without ever checking.</div></div></div>"""))

    # 14 results
    write(s, slide("""
      <h2>How fast can it go?</h2>
      <table>
      <tr><th>model</th><th>max pass speed</th><th>planner inset</th><th>peak a<sub>y</sub></th>
          <th>grip used</th><th>peak β</th><th>margin</th></tr>
      <tr><td><b>double-track</b></td><td class="num"><b>$(round(D.speed,digits=2)) m/s ($(round(Int,3.6*D.speed)) km/h)</b></td>
          <td class="num">$(D.inset) m</td><td class="num">$(round(maximum(abs,D.res.traj.ay),digits=2))</td>
          <td class="num">$(round(Int,100*maximum(abs,D.res.traj.ay)/9.0)) %</td>
          <td class="num">$(round(rad2deg(maximum(abs,D.res.traj.beta)),digits=2))°</td>
          <td class="num">$(round(D.res.margin,digits=3)) m</td></tr>
      <tr><td>bicycle</td><td class="num">$(round(B.speed,digits=2)) m/s ($(round(Int,3.6*B.speed)) km/h)</td>
          <td class="num">$(B.inset) m</td><td class="num">$(round(maximum(abs,B.res.traj.ay),digits=2))</td>
          <td class="num">$(round(Int,100*maximum(abs,B.res.traj.ay)/9.0)) %</td>
          <td class="num">$(round(rad2deg(maximum(abs,B.res.traj.beta)),digits=2))°</td>
          <td class="num">$(round(B.res.margin,digits=3)) m</td></tr>
      </table>
      <p>The bicycle reads <b>$(round(100*(B.speed-D.speed)/D.speed,digits=1)) % optimistic</b>,
      consistent with the open-loop finding that it overstates limit grip by about 6 %.</p>
      <div class="cols"><div>
      <h3>The driver is not the bottleneck</h3>
      <p>Which is what "reach the highest possible speed" actually demands. Sweeping the
      MPC weights and horizon, each configuration re-optimised over the planner inset,
      moves the bisected speed by under <b>3 %</b> — a 16× range on tracking weight, a 6×
      range on the rate penalty, horizons from 0.6 s to 1.6 s. The limit belongs to the
      vehicle and the course.</p>
      <div class="warn"><b>But that insensitivity hid a real fault.</b> Max speed cannot
      see steering quality: the car stays in the corridor whether the wheel moves smoothly
      or thrashes. Tuning on speed alone left a cost function that spent ±110° of steering
      to chase millimetres. See the next slide.</div></div>
      <div><h3>Pass criterion</h3><ul>
        <li>All four <b>body corners</b> inside the corridor — cones are struck by bodywork,
        not by the centre of gravity</li>
        <li>No spin (|β| &lt; 20°)</li>
        <li>Run stays inside the model's forward-motion validity envelope</li>
        <li>Entry speed bisected to 0.05 m/s, with both brackets checked</li></ul>
      <div class="note">The planner inset is part of the driver: it trades corridor margin
      against path curvature, both monotone and opposed, so there is an interior optimum.
      Reporting any single inset would understate the car.</div></div></div>"""))

    # 14b steering diagnosis
    write(s, slide("""
      <h2>Diagnosing a steering trace that looked wrong</h2>
      <p>The commanded steering was oscillatory and visibly larger than the path needs.
      Four hypotheses, eliminated in order — each test designed so a negative result was
      as informative as a positive one.</p>
      <table>
      <tr><th>Hypothesis</th><th>Test</th><th>Result</th></tr>
      <tr><td>Rough reference curvature</td>
          <td>swap the kinked QP reference for a C² analytic one</td>
          <td class="bad">not the cause — 3× lower peak steering <i>rate</i>, but the
          oscillation stayed</td></tr>
      <tr><td>Tyre saturation / model mismatch</td>
          <td>sweep speed from 22 % to 94 % grip</td>
          <td class="bad">not the cause — oscillation present at 22 % grip too</td></tr>
      <tr><td>Penalising yaw rate <code>r</code> instead of yaw-rate <i>error</i></td>
          <td>set <code>q_r</code> = 0.3, 0.1, 0</td>
          <td class="bad">not the cause — curves essentially identical</td></tr>
      <tr><td>Finite horizon, no terminal cost</td>
          <td>horizon 1 s → 7 s</td>
          <td class="bad">not the cause — all horizons give the <i>same</i> curve</td></tr>
      <tr><td><b>Badly scaled cost</b></td>
          <td>check condensing and objective against explicit evaluation</td>
          <td class="good">both exact (2×10⁻¹⁶) — so the ugly trace <b>was</b> the true
          optimum</td></tr>
      </table>
      <div class="key"><b>Root cause.</b> With <code>q_ey = 48</code> per m² and
      <code>R = 0.5</code> per rad², a 0.1 m lateral error costs 0.48 while a 0.1 rad
      steering input costs 0.005 — steering was ~100× cheaper than error. The optimiser
      did exactly what it was asked: spend unlimited steering to drive the error to
      millimetres. Nothing was broken; the question was wrong.</div>
      <div class="cols"><div><h3>Fix 1 — scale the cost</h3>
      <p><code>R = 50</code> balances a 0.1 m error against a 2.5° road-wheel input.
      Peak steering 109° → 66°, rms rate 389 → 183 deg/s, <b>speed unchanged</b>.</p></div>
      <div><h3>Fix 2 — re-linearise (LTV)</h3>
      <p>At 31 m/s the real secant stiffness is 0.50 of nominal at the front and
      <b>0.25</b> at the rear, so the MPC believed the car four times more responsive than
      it is. Re-linearising at the operating point cut tracking error 75 → 28 mm and
      <i>raised</i> the limit from 31.7 to 32.5 m/s.</p></div></div>"""))

    # 15 dlc figure
    write(s, slide("""
      <h2>The manoeuvre</h2>
      <figure><img src="$(img(joinpath(@__DIR__,"dlc_mpc.png")))"><figcaption>
      After rescaling the cost and re-linearising: peak steering
      $(round(Int, rad2deg(maximum(abs, D.res.traj.delta)*STEER_RATIO)))° and peak yaw
      ±$(round(rad2deg(maximum(abs, D.res.traj.psi)), digits=1))°. The bottom row is
      against <b>time</b>, not station, which is what makes the two cars comparable as
      driver workload.</figcaption></figure>"""))

    # 16 3D DLC
    write(s, slide("""
      <h2>The manoeuvre in 3D</h2>
      <figure><img src="$(img(FIG_SCENE))" style="max-height:44vh"><figcaption>
      Still rendered from the same meshes the interactive scene uses.</figcaption></figure>
      <div class="cards">
        <a class="card" href="dlc_3d_follow.html"><div class="t">At the limit — $(round(D.speed,digits=1)) m/s</div>
          <div class="d">Animated follow camera. Zero cones struck, 1 mm of margin.</div></a>
        <a class="card" href="dlc_3d_fail.html"><div class="t">Over the limit — 33.2 m/s</div>
          <div class="d">Same car, same reference, 1.5 m/s faster. Three cones turn red.</div></a>
        <a class="card" href="dlc_3d_overview.html"><div class="t">Whole course</div>
          <div class="d">All 125 m with ghost cars and the full cone layout.</div></a>
      </div>
      <p>Each scene carries a <b>steering-wheel gauge</b> (16:1 ratio, a readout — the model
      has no steering column) whose orientation stays fixed in world axes while its position
      follows the car, so the rim shows steering alone and not steering plus yaw. Beside the
      3D view, live 2D panels trace steering-wheel angle and yaw angle with a cursor locked
      to the animation.</p>"""))

    # 17 bugs
    write(s, slide("""
      <h2>What validation caught</h2>
      <p>Every one of these produced plausible output. None would have been found by
      looking at a return code.</p>
      <table>
      <tr><th>Defect</th><th>Symptom</th><th>Root cause</th></tr>
      <tr><td>Anti-windup units</td><td>speed settled 0.54 m/s low</td>
          <td>(F<sub>x</sub>−F<sub>raw</sub>)/T<sub>t</sub> is N/s added to an m/s balance;
          needs the k<sub>i</sub> factor</td></tr>
      <tr><td>QP linear term</td><td>dimension mismatch</td>
          <td>S<sub>u</sub>ᵀQ̄ is N×nN — the transpose was wrong</td></tr>
      <tr><td>QP rate penalty</td><td>silently wrong optimum</td>
          <td>D<sub>m</sub>ᵀe₁ = e₁, not the first column of D<sub>m</sub></td></tr>
      <tr><td>Coordinate descent</td><td>11 % above optimum, no constraint active</td>
          <td>cond(H) ≈ 5×10³ after condensing</td></tr>
      <tr><td>Active set early return</td><td>all-pinned case never checked</td>
          <td><code>isempty(free)</code> returned before the dual test</td></tr>
      <tr><td>Cone placement</td><td>pass run "struck" 11 cones, fail run 6</td>
          <td>cone centres on the boundary, not tangent to it</td></tr>
      <tr><td>Infeasible corridor</td><td>κ = 0.59, R = 1.7 m</td>
          <td>inset inverted the gate; QP had no feasible point</td></tr>
      <tr><td>Inert play button</td><td>control present, nothing happened</td>
          <td>transport controls emitted on a figure with no frames</td></tr>
      <tr><td>Badly scaled MPC cost</td><td>±110° of steering to hold millimetres</td>
          <td>tuned on max speed, which is blind to steering effort</td></tr>
      <tr><td>Constant tyre stiffness in the MPC</td><td>tracking error 75 mm at the limit</td>
          <td>real secant stiffness falls to 0.25 of nominal at the rear</td></tr>
      </table>
      <div class="key">The pattern: <b>the thing that found each bug was a number computed a
      second way</b> — an analytic trim force, a KKT residual, a known identity, a
      cross-check between two tests that must agree.</div>"""))

    # 18 limitations
    write(s, slide("""
      <h2>What this model is not</h2>
      <div class="cols"><div><h3>Structural limits</h3><ul>
        <li><b>Three degrees of freedom.</b> No roll, pitch, heave or suspension travel.
        Load transfer is quasi-static with a lag. Body attitude in the 3D views is a
        reconstruction, not a state.</li>
        <li><b>No wheel-spin dynamics.</b> Longitudinal force is commanded; combined slip
        is handled by the friction ellipse. ABS or traction control would need 4 more states.</li>
        <li><b>Forward motion only</b>, enforced by assertion.</li>
        <li><b>No steering column.</b> The steering-wheel angle is a 16:1 readout.</li>
      </ul></div>
      <div><h3>Choices worth challenging</h3><ul>
        <li><b>Gate geometry is parameterised, not certified.</b> Defaults approximate
        ISO 3888-1; I deliberately did not assert standard dimensions from memory.</li>
        <li><b>Understeer comes from a per-axle stiffness split</b> (B<sub>f</sub> = 8,
        B<sub>r</sub> = 10, giving 0.83 deg/g). With identical tyres the model is nearly
        neutral, because load sensitivity alone almost exactly cancels the weight
        distribution — real understeer comes mostly from suspension kinematics this model
        does not have, so it is lumped into the axle characteristic.</li>
        <li><b>Reported speeds depend on the planner inset</b>, which is optimised over.
        It is the best this driver achieves, not a certified ISO result.</li>
        <li><b>Cones are discrete at 3 m</b>, so a shallow incursion between two cones is a
        negative margin with no cone struck. The margin plot is the strict test.</li>
      </ul></div></div>"""))

    # 19 reproduce
    write(s, slide("""
      <h2>Reproducing all of it</h2>
      <pre><span class="c"># step-steer comparison: figure + steady-state table</span>
julia --project=. scripts/compare_vehicle_models.jl

<span class="c"># closed-loop DLC: max-speed search, driver sensitivity, figure</span>
julia --project=. scripts/dlc_mpc.jl

<span class="c"># interactive 3D scenes</span>
julia --project=. scripts/visualize_vehicle_3d.jl
julia --project=. scripts/visualize_dlc_3d.jl

<span class="c"># this deck</span>
julia --project=. scripts/presentation.jl</pre>
      <div class="cols"><div><h3>Dyad analyses</h3>
      <pre><span class="k">using</span> TestVehicleDynamics, DyadInterface
res = TireSweep()
res = BicycleStepSteer()
res = DoubleTrackStepSteer()
sol = res.sol
m   = symbolic_container(res)</pre></div>
      <div><h3>Dependencies</h3>
      <p>Only <code>BlockComponents</code> was added to <code>Project.toml</code>.
      <code>Plots</code>, <code>PlotlyBase</code> and <code>FFMPEG</code> ship with the Dyad
      runtime, so the visualisations need no installs.</p>
      <div class="note"><b>PlotlyKaleido does not work in this runtime</b> — its binary
      artifact points at a CI path that does not exist locally. Static export of the
      interactive scenes is unavailable; the still on the 3D slide is rendered directly
      from the meshes instead.</div></div></div>"""))

    body = String(take!(s))
    return """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>3-DOF Vehicle Dynamics in Dyad</title><style>$CSS</style></head>
<body><div id="deck">
$body<div id="bar"></div>
<div id="nav"><button onclick="go(i-1)">←</button><span id="num"></span><button onclick="go(i+1)">→</button></div>
</div><script>$JS</script></body></html>"""
end

function main()
    @info "generating figures"
    tyre_figure()
    scene_still()
    @info "running the studies for live numbers"
    results = Base.invokelatest(() -> begin
        course = DLCCourse(); mpc = MPCParams()
        out = Dict{Symbol,Any}()
        for which in (:doubletrack, :bicycle)
            P = build_plant(which)
            b = best_inset(P; course, mpc)
            out[which] = (; b..., res = evaluate(P, b.speed; course, path = b.path, mpc))
        end
        out
    end)
    open(DECK, "w") do io; write(io, build_deck(results)); end
    @info "deck written" DECK size_kb=round(filesize(DECK)/1024, digits=1)
    return DECK
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
