"""
Section 2: Physical Simulation Visualization
Animated visualization of a damped harmonic oscillator with phase portrait and displacement history. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec

# --- 1. Physics Engine ---
# Parameters
m = 2.0       # Mass (kg)
k = 40.0      # Spring constant (N/m)
c = 1.5       # Damping (Lowered slightly to show more spirals)
g = 9.81
rest_length = 1.5
y_eq_offset = (m * g) / k
y_equilibrium = rest_length + y_eq_offset

# Initial State
y0 = y_equilibrium + 1.2  # Pull down 1.2m
v0 = 0.0

# Simulation Settings
dt = 0.04
t_max = 15.0
steps = int(t_max / dt)
time = np.linspace(0, t_max, steps)

# Pre-calculate Physics (Euler integration)
y = np.zeros(steps)
v = np.zeros(steps)
y[0] = y0
v[0] = v0

for i in range(steps - 1):
    f_g = m * g
    stretch = y[i] - rest_length
    f_s = -k * stretch
    f_d = -c * v[i]
    a = (f_g + f_s + f_d) / m
    v[i+1] = v[i] + a * dt
    y[i+1] = y[i] + v[i+1] * dt

# --- 2. Visualization Design ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, width_ratios=[1, 1.5])

# Color Palette
col_bg = '#1e1e1e'
col_mass = '#00f2ff'  # Cyan Neon
col_spring = '#ffffff'
col_trail = '#00f2ff'
col_force_g = '#555555'  # Grey
col_force_s = '#ff0055'  # Neon Red
col_force_d = '#ffe600'  # Neon Yellow

fig.patch.set_facecolor(col_bg)

# --- A. Main Simulation View ---
ax_sim = fig.add_subplot(gs[: , 0])
ax_sim.set_facecolor(col_bg)
ax_sim.invert_yaxis()
ax_sim.set_xlim(-1.5, 1.5)
ax_sim.set_ylim(y0 + 0.5, -0.5)
ax_sim.axis('off')
ax_sim.set_title("PHYSICAL SIMULATION VIEW", color='white', fontsize=10, pad=20, loc='left')

# Reference Lines
ax_sim.axhline(y_equilibrium, color='#444444', linestyle='--', alpha=0.5, lw=1)
ax_sim.text(-1.4, y_equilibrium, "EQUILIBRIUM LINE", color='#444444', fontsize=8, va='bottom')

# Elements
spring_line, = ax_sim.plot([], [], color=col_spring, lw=2)
damper_cyl, = ax_sim.plot([], [], color='#888888', lw=2)
damper_pis, = ax_sim.plot([], [], color='#aaaaaa', lw=2)
mass_rect = Rectangle((-0.4, 0), 0.8, 0.6, color=col_mass, alpha=0.8, ec='white', zorder=10)
ax_sim.add_patch(mass_rect)

# Ghost Trail
trail_line, = ax_sim.plot([], [], color=col_trail, alpha=0.3, lw=1)

# Arrows (Vectors)
arrow_style = '-|>,head_length=0.4,head_width=0.2'
vec_g = FancyArrowPatch((0,0), (0,0), arrowstyle=arrow_style, color=col_force_g, lw=2)
vec_s = FancyArrowPatch((0,0), (0,0), arrowstyle=arrow_style, color=col_force_s, lw=2)
vec_d = FancyArrowPatch((0,0), (0,0), arrowstyle=arrow_style, color=col_force_d, lw=2)
ax_sim.add_patch(vec_g)
ax_sim.add_patch(vec_s)
ax_sim.add_patch(vec_d)

# Labels for Arrows
lbl_g = ax_sim.text(0, 0, "g", color=col_force_g, fontsize=9, ha='right')
lbl_s = ax_sim.text(0, 0, "Fs", color=col_force_s, fontsize=9, ha='right')
lbl_d = ax_sim.text(0, 0, "Fd", color=col_force_d, fontsize=9, ha='left')

# --- B. Phase Portrait ---
ax_phase = fig.add_subplot(gs[0, 1])
ax_phase.set_facecolor('#252525')
ax_phase.set_title("PHASE PORTRAIT (Velocity vs Position)", color='#aaaaaa', fontsize=9)
ax_phase.set_xlabel("Position (m)", color='#666666', fontsize=8)
ax_phase.set_ylabel("Velocity (m/s)", color='#666666', fontsize=8)
ax_phase.grid(True, color='#333333')
ax_phase.spines['bottom'].set_color('#555555')
ax_phase.spines['left'].set_color('#555555')
ax_phase.spines['top'].set_visible(False)
ax_phase.spines['right'].set_visible(False)
ax_phase.tick_params(axis='x', colors='#888888')
ax_phase.tick_params(axis='y', colors='#888888')

phase_line, = ax_phase.plot([], [], color=col_mass, lw=1.5, alpha=0.8)
phase_dot, = ax_phase.plot([], [], 'o', color='white', markersize=4)

# --- C. Time Series ---
ax_time = fig.add_subplot(gs[1, 1])
ax_time.set_facecolor('#252525')
ax_time.set_title("DISPLACEMENT HISTORY", color='#aaaaaa', fontsize=9)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(min(y), max(y))
ax_time.grid(True, color='#333333')
ax_time.spines['bottom'].set_color('#555555')
ax_time.spines['left'].set_color('#555555')
ax_time.spines['top'].set_visible(False)
ax_time.spines['right'].set_visible(False)
ax_time.tick_params(colors='#888888')

time_line, = ax_time.plot([], [], color=col_spring, lw=1.5)
time_dot, = ax_time.plot([], [], 'o', color=col_mass, markersize=5)

# --- Animation Update Function ---
def update(frame):
    # Data extraction
    cy = y[frame]
    cv = v[frame]
    t = time[frame]

    # 1. Update Sim Objects
    mass_rect.set_y(cy - 0.3)

    # Spring
    sy = np.linspace(0, cy-0.3, 50)
    sx = 0.2 * np.sin(2 * np.pi * 8 * sy / (cy-0.3))
    spring_line.set_data(sx, sy)

    # Damper
    d_offset = -0.6
    damper_cyl.set_data([d_offset, d_offset], [0, rest_length*0.7])
    damper_pis.set_data([d_offset, d_offset], [rest_length*0.6, cy-0.3])

    # Trail
    trail_len = 100
    start = max(0, frame-trail_len)
    trail_line.set_data(np.zeros(frame-start), y[start:frame])

    # 2. Update Force Vectors
    f_scale = 0.015

    # Gravity (Down)
    fg_len = (m*g) * f_scale
    vec_g.set_positions((0, cy), (0, cy + fg_len))
    lbl_g.set_position((-0.1, cy + fg_len/2))

    # Spring (Up/Down depending on stretch)
    fs_val = -k * (cy - rest_length)
    fs_len = fs_val * f_scale
    vec_s.set_positions((0, cy-0.3), (0, cy-0.3 - fs_len))
    lbl_s.set_position((-0.1, cy-0.3 - fs_len - 0.1))

    # Damping (Opposite to Velocity)
    fd_val = -c * cv
    fd_len = fd_val * f_scale
    vec_d.set_positions((0.5, cy), (0.5, cy - fd_len))
    lbl_d.set_position((0.6, cy - fd_len/2))

    # 3. Update Phase Plot
    phase_line.set_data(y[:frame], v[:frame])
    phase_dot.set_data([cy], [cv])

    # 4. Update Time Plot
    time_line.set_data(time[:frame], y[:frame])
    time_dot.set_data([t], [cy])

    return spring_line, mass_rect, vec_g, vec_s, vec_d, phase_line, phase_dot, time_line, time_dot, trail_line, damper_cyl, damper_pis, lbl_g, lbl_s, lbl_d

# Run Animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=True)

# Save
ani.save('pro_damped_dashboard.gif', writer='pillow', fps=30)
# plt.close()