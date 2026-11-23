"""
BIT 2118 Scientific Computing - Task 2
NumPy Arrays and Numerical Computation (application: IoT sensor readings)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# A. MAIN CASE: temperature readings from an IoT sensor
# ------------------------------------------------------------------
print("=== A. MAIN CASE: IoT temperature sensor ===")
temps = np.array([20, 22, 24, 26], dtype=float)      # worksheet example
print("temperatures (C)  :", temps)
print("squares x**2      :", temps ** 2)             # one operation on every element
print("mean (np.mean)    :", np.mean(temps))         # (20+22+24+26)/4 = 23

hours = np.arange(24)                                # one reading per hour
readings = 22 + 4 * np.sin((hours - 9) * np.pi / 12) # simulated 24-hour profile
readings_f = readings * 9 / 5 + 32                   # Celsius -> Fahrenheit (vectorised)
limit = 25.5
alarm_hours = hours[readings > limit]                # Boolean mask = alarm hours
print("24h mean/min/max  : %.2f / %.2f / %.2f C" %
      (readings.mean(), readings.min(), readings.max()))
print("first 3 readings F:", np.round(readings_f[:3], 2))
print("hours above %.1f C: %s" % (limit, alarm_hours))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: one faulty reading (spike) at hour 12
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: faulty spike of 85 C at hour 12 ===")
faulty = readings.copy()
faulty[12] = 85.0
print("mean   before -> after : %.2f -> %.2f" % (readings.mean(), faulty.mean()))
print("median before -> after : %.2f -> %.2f" %
      (np.median(readings), np.median(faulty)))
print("std    before -> after : %.2f -> %.2f" % (readings.std(), faulty.std()))
valid = faulty[(faulty > -10) & (faulty < 45)]       # simple validity filter
print("mean after filtering   : %.2f  (%d valid readings)" % (valid.mean(), valid.size))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(hours, readings, "o-", label="normal readings")
ax.plot(hours, faulty, "s--", color="tab:red", alpha=0.7, label="with faulty spike")
ax.axhline(limit, color="gray", ls=":", label="alarm limit 25.5 C")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Temperature (C)")
ax.set_title("Task 2: IoT temperature readings")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task02_sensor.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: soil-moisture sensors on 6 farm plots
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: soil moisture and irrigation ===")
moisture = np.array([32, 28, 35, 18, 22, 30], dtype=float)   # % volumetric
area = np.array([120, 80, 150, 100, 90, 110], dtype=float)   # m^2 per plot
target = 30.0                                                # % desired moisture
litres_per_pct_m2 = 5.0                                      # assumption
deficit = np.clip(target - moisture, 0, None)                # no negative deficit
water = deficit * area * litres_per_pct_m2
print("moisture (%)      :", moisture)
print("plots to irrigate :", np.where(moisture < target)[0] + 1)
print("water needed (L)  :", water)
print("total water (L)   : %.0f" % water.sum())
print("farm mean moisture: %.2f %%  (area weighted: %.2f %%)" %
      (moisture.mean(), np.average(moisture, weights=area)))

fig, ax = plt.subplots(figsize=(7, 3.8))
colors = np.where(moisture < target, "tab:red", "tab:green")
ax.bar(np.arange(1, 7), moisture, color=colors)
ax.axhline(target, color="k", ls="--", label="target 30 %")
ax.set_xlabel("Plot number")
ax.set_ylabel("Soil moisture (%)")
ax.set_title("Task 2 (new case): plots below target need irrigation")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task02_soil.png", dpi=150)
plt.close(fig)
