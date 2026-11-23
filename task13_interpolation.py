"""
BIT 2118 Scientific Computing - Task 13
Interpolation (scipy.interpolate: interp1d, CubicSpline)
"""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def thermistor(T):
    """'True' resistance (kOhm) of a 10 kOhm NTC thermistor, B = 3950 K."""
    return 10.0 * np.exp(3950.0 * (1 / (T + 273.15) - 1 / 298.15))


# ------------------------------------------------------------------
# A. MAIN CASE: sensor calibration table, estimate between measured points
# ------------------------------------------------------------------
print("=== A. MAIN CASE: thermistor calibration ===")
T_cal = np.array([0, 10, 20, 30, 40, 50], dtype=float)       # measured temperatures
R_cal = thermistor(T_cal)                                    # measured resistances
lin = interp1d(T_cal, R_cal, kind="linear")
cub = interp1d(T_cal, R_cal, kind="cubic")
spl = CubicSpline(T_cal, R_cal)
print("calibration table (T C, R kOhm):")
for t, r in zip(T_cal, R_cal):
    print("   %4.0f  %8.4f" % (t, r))
for Tq in [25.0, 35.0]:
    true = thermistor(Tq)
    print("T = %.0f C: true %.4f | linear %.4f (err %.4f) | cubic %.4f (err %.4f)"
          " | CubicSpline %.4f" %
          (Tq, true, lin(Tq), abs(lin(Tq) - true), cub(Tq), abs(cub(Tq) - true),
           spl(Tq)))

Tf = np.linspace(0, 50, 300)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(Tf, thermistor(Tf), "k-", lw=1, label="true curve")
ax.plot(Tf, lin(Tf), "--", label="linear")
ax.plot(Tf, spl(Tf), "-.", label="cubic spline")
ax.plot(T_cal, R_cal, "ro", label="calibration points")
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("Resistance (kOhm)")
ax.set_title("Task 13: interpolating a sensor calibration table")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task13_calibration.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: one measurement is missing (T = 30 C), estimate it
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: missing measurement at 30 C ===")
keep = T_cal != 30
Tm, Rm = T_cal[keep], R_cal[keep]
true30 = thermistor(30.0)
lin_m = interp1d(Tm, Rm, kind="linear")(30.0)
cub_m = CubicSpline(Tm, Rm)(30.0)
print("true R(30) = %.4f kOhm" % true30)
print("linear estimate = %.4f (error %.4f)" % (lin_m, abs(lin_m - true30)))
print("cubic  estimate = %.4f (error %.4f)" % (cub_m, abs(cub_m - true30)))
print("the gap is now 20 C wide, so the errors grow compared with the 10 C case")

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: electricity load measured every 3 hours,
#    estimate hourly values (resampling). Values are illustrative.
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: resampling a 3-hourly electricity load (MW) ===")
hrs = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24], dtype=float)
load = np.array([1150, 1080, 1250, 1560, 1500, 1480, 1860, 1700, 1160], dtype=float)
hourly = np.arange(0, 25)
lin_l = interp1d(hrs, load)(hourly)
spl_l = CubicSpline(hrs, load)(hourly)
print("estimated load at 19:30 -> linear %.1f MW | cubic spline %.1f MW" %
      (interp1d(hrs, load)(19.5), CubicSpline(hrs, load)(19.5)))
print("estimated evening peak (cubic spline): %.1f MW at hour %d" %
      (spl_l.max(), hourly[spl_l.argmax()]))
print("hourly estimates 17..20 (cubic):", np.round(spl_l[17:21], 1))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(hourly, lin_l, "--", label="linear")
ax.plot(hourly, spl_l, "-", label="cubic spline")
ax.plot(hrs, load, "ro", label="measured")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Load (MW)")
ax.set_title("Task 13 (new case): hourly load from 3-hourly data")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task13_load.png", dpi=150)
plt.close(fig)
