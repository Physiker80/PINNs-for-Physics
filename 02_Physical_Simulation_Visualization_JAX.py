"""
Section 2: Physical Simulation Visualization (JAX Version) - Enhanced GUI
Animated visualization of a damped harmonic oscillator with phase portrait and displacement history.
Rebuilt using JAX for high-performance numerical computing with improved visuals.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# --- 1. Physics Engine (JAX) ---
# Parameters
m = 2.0       # Mass (kg)
k = 40.0      # Spring constant (N/m)
c = 1.5       # Damping coefficient
g = 9.81      # Gravity (m/s^2)
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
time = jnp.linspace(0, t_max, steps)

# JIT-compiled physics step function
@jit
def physics_step(state, _):
    """Single step of Euler integration for damped harmonic oscillator."""
    y_curr, v_curr = state
    
    # Forces
    f_gravity = m * g
    stretch = y_curr - rest_length
    f_spring = -k * stretch
    f_damping = -c * v_curr
    
    # Acceleration
    a = (f_gravity + f_spring + f_damping) / m
    
    # Euler integration
    v_next = v_curr + a * dt
    y_next = y_curr + v_next * dt
    
    return (y_next, v_next), (y_next, v_next)

# Run simulation using lax.scan (JAX's efficient loop)
@jit
def run_simulation(y_init, v_init):
    """Run the full simulation using JAX's scan for efficiency."""
    initial_state = (y_init, v_init)
    _, (y_history, v_history) = lax.scan(physics_step, initial_state, None, length=steps-1)
    
    # Prepend initial conditions
    y_full = jnp.concatenate([jnp.array([y_init]), y_history])
    v_full = jnp.concatenate([jnp.array([v_init]), v_history])
    
    return y_full, v_full

# Execute simulation
print("Running JAX simulation...")
y, v = run_simulation(y0, v0)

# Convert to numpy for matplotlib
y = np.array(y)
v = np.array(v)
time_np = np.array(time)

print(f"Simulation complete. Shape: {y.shape}, Device: {jax.devices()[0]}")

# --- 2. Enhanced Visualization Design ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 3, width_ratios=[1.2, 1, 0.8], height_ratios=[1, 1], 
              hspace=0.3, wspace=0.3)

# Enhanced Color Palette
col_bg = '#0a0a0f'
col_panel = '#12121a'
col_mass = '#00f2ff'       # Cyan Neon
col_mass_glow = '#00f2ff40'
col_spring = '#50fa7b'     # Green Neon
col_spring_coil = '#50fa7b'
col_trail = '#00f2ff'
col_force_g = '#ff79c6'    # Pink - Gravity
col_force_s = '#ff5555'    # Red - Spring
col_force_d = '#ffb86c'    # Orange - Damping
col_force_net = '#8be9fd'  # Cyan - Net force
col_anchor = '#6272a4'
col_text = '#f8f8f2'
col_text_dim = '#6272a4'

fig.patch.set_facecolor(col_bg)

# --- A. Main Simulation View ---
ax_sim = fig.add_subplot(gs[:, 0])
ax_sim.set_facecolor(col_bg)
ax_sim.invert_yaxis()
ax_sim.set_xlim(-1.8, 1.8)
ax_sim.set_ylim(y0 + 0.8, -0.8)
ax_sim.axis('off')

# Title with glow effect
ax_sim.text(-1.7, -0.6, "⚡ SPRING-MASS SYSTEM", color=col_mass, fontsize=12, 
            fontweight='bold', va='top', ha='left')
ax_sim.text(-1.7, -0.4, "Damped Harmonic Oscillator", color=col_text_dim, fontsize=9, 
            va='top', ha='left')

# Anchor/ceiling
anchor_rect = FancyBboxPatch((-0.8, -0.15), 1.6, 0.15, 
                              boxstyle="round,pad=0.02", 
                              facecolor=col_anchor, edgecolor='#8be9fd', linewidth=2)
ax_sim.add_patch(anchor_rect)
# Anchor hatching lines
for i in range(-8, 9, 2):
    ax_sim.plot([i*0.08, i*0.08 + 0.1], [-0.15, -0.25], color='#44475a', lw=1)

# Reference Lines with labels
ax_sim.axhline(y_equilibrium, color='#44475a', linestyle='--', alpha=0.7, lw=1.5)
ax_sim.axhline(rest_length, color='#6272a4', linestyle=':', alpha=0.5, lw=1)
eq_label = ax_sim.text(1.5, y_equilibrium, f"EQ: {y_equilibrium:.2f}m", color='#44475a', 
                       fontsize=8, va='center', ha='left',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=col_bg, edgecolor='#44475a', alpha=0.8))
rest_label = ax_sim.text(1.5, rest_length, f"REST: {rest_length:.2f}m", color='#6272a4', 
                         fontsize=8, va='center', ha='left',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor=col_bg, edgecolor='#6272a4', alpha=0.8))

# Spring coil elements (will be updated in animation)
spring_line, = ax_sim.plot([], [], color=col_spring_coil, lw=2.5, solid_capstyle='round')
spring_glow, = ax_sim.plot([], [], color=col_spring_coil, lw=6, alpha=0.2)

# Damper visualization
damper_body = FancyBboxPatch((-0.72, 0), 0.24, 0.8, boxstyle="round,pad=0.02",
                              facecolor='#282a36', edgecolor='#6272a4', linewidth=1.5)
ax_sim.add_patch(damper_body)
damper_piston, = ax_sim.plot([], [], color='#bd93f9', lw=4, solid_capstyle='round')
damper_rod, = ax_sim.plot([], [], color='#6272a4', lw=2)

# Mass block with glow
mass_glow = Rectangle((-0.45, 0), 0.9, 0.7, color=col_mass_glow, alpha=0.3, zorder=9)
ax_sim.add_patch(mass_glow)
mass_rect = FancyBboxPatch((-0.4, 0), 0.8, 0.6, boxstyle="round,pad=0.03",
                            facecolor=col_mass, edgecolor='white', linewidth=2, zorder=10)
ax_sim.add_patch(mass_rect)
mass_label = ax_sim.text(0, 0, f"{m} kg", color=col_bg, fontsize=10, fontweight='bold',
                         ha='center', va='center', zorder=11)

# Ghost Trail with gradient
trail_line, = ax_sim.plot([], [], color=col_trail, alpha=0.4, lw=2)

# Force Arrows with enhanced styling
arrow_style_thick = '-|>,head_length=0.15,head_width=0.1'

# Create force vectors
vec_g = FancyArrowPatch((0, 0), (0, 0), arrowstyle=arrow_style_thick, 
                         color=col_force_g, lw=3, mutation_scale=15)
vec_s = FancyArrowPatch((0, 0), (0, 0), arrowstyle=arrow_style_thick, 
                         color=col_force_s, lw=3, mutation_scale=15)
vec_d = FancyArrowPatch((0, 0), (0, 0), arrowstyle=arrow_style_thick, 
                         color=col_force_d, lw=3, mutation_scale=15)
vec_net = FancyArrowPatch((0, 0), (0, 0), arrowstyle=arrow_style_thick, 
                           color=col_force_net, lw=4, mutation_scale=18)

ax_sim.add_patch(vec_g)
ax_sim.add_patch(vec_s)
ax_sim.add_patch(vec_d)
ax_sim.add_patch(vec_net)

# Force labels with values
lbl_g = ax_sim.text(0, 0, "", color=col_force_g, fontsize=9, fontweight='bold', ha='left')
lbl_s = ax_sim.text(0, 0, "", color=col_force_s, fontsize=9, fontweight='bold', ha='right')
lbl_d = ax_sim.text(0, 0, "", color=col_force_d, fontsize=9, fontweight='bold', ha='left')
lbl_net = ax_sim.text(0, 0, "", color=col_force_net, fontsize=9, fontweight='bold', ha='center')

# --- Force Legend Panel ---
legend_y = y0 + 0.4
ax_sim.text(-1.7, legend_y, "FORCES", color=col_text, fontsize=9, fontweight='bold')
ax_sim.plot([-1.7, -1.5], [legend_y + 0.2, legend_y + 0.2], color=col_force_g, lw=3)
ax_sim.text(-1.45, legend_y + 0.2, "Gravity (mg)", color=col_force_g, fontsize=8, va='center')
ax_sim.plot([-1.7, -1.5], [legend_y + 0.35, legend_y + 0.35], color=col_force_s, lw=3)
ax_sim.text(-1.45, legend_y + 0.35, "Spring (-kx)", color=col_force_s, fontsize=8, va='center')
ax_sim.plot([-1.7, -1.5], [legend_y + 0.5, legend_y + 0.5], color=col_force_d, lw=3)
ax_sim.text(-1.45, legend_y + 0.5, "Damping (-cv)", color=col_force_d, fontsize=8, va='center')
ax_sim.plot([-1.7, -1.5], [legend_y + 0.65, legend_y + 0.65], color=col_force_net, lw=4)
ax_sim.text(-1.45, legend_y + 0.65, "Net Force", color=col_force_net, fontsize=8, va='center')

# --- B. Phase Portrait ---
ax_phase = fig.add_subplot(gs[0, 1])
ax_phase.set_facecolor(col_panel)
ax_phase.set_title("◉ PHASE PORTRAIT", color=col_text, fontsize=10, fontweight='bold', loc='left', pad=10)
ax_phase.set_xlabel("Position (m)", color=col_text_dim, fontsize=9)
ax_phase.set_ylabel("Velocity (m/s)", color=col_text_dim, fontsize=9)
ax_phase.grid(True, color='#282a36', linestyle='-', alpha=0.5)
ax_phase.set_xlim(min(y) - 0.1, max(y) + 0.1)
ax_phase.set_ylim(min(v) - 0.5, max(v) + 0.5)
for spine in ax_phase.spines.values():
    spine.set_color('#44475a')
    spine.set_linewidth(1.5)
ax_phase.tick_params(axis='both', colors='#6272a4', labelsize=8)

# Equilibrium marker
ax_phase.axvline(y_equilibrium, color='#44475a', linestyle='--', alpha=0.5, lw=1)
ax_phase.axhline(0, color='#44475a', linestyle='--', alpha=0.5, lw=1)

phase_trail, = ax_phase.plot([], [], color=col_mass, lw=1, alpha=0.3)
phase_line, = ax_phase.plot([], [], color=col_mass, lw=2, alpha=0.9)
phase_dot, = ax_phase.plot([], [], 'o', color='white', markersize=8, zorder=10)
phase_glow, = ax_phase.plot([], [], 'o', color=col_mass, markersize=14, alpha=0.3, zorder=9)

# --- C. Time Series ---
ax_time = fig.add_subplot(gs[1, 1])
ax_time.set_facecolor(col_panel)
ax_time.set_title("◉ DISPLACEMENT vs TIME", color=col_text, fontsize=10, fontweight='bold', loc='left', pad=10)
ax_time.set_xlabel("Time (s)", color=col_text_dim, fontsize=9)
ax_time.set_ylabel("Position (m)", color=col_text_dim, fontsize=9)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(min(y) - 0.1, max(y) + 0.1)
ax_time.grid(True, color='#282a36', linestyle='-', alpha=0.5)
for spine in ax_time.spines.values():
    spine.set_color('#44475a')
    spine.set_linewidth(1.5)
ax_time.tick_params(axis='both', colors='#6272a4', labelsize=8)

# Equilibrium line
ax_time.axhline(y_equilibrium, color='#44475a', linestyle='--', alpha=0.5, lw=1)

time_fill = ax_time.fill_between([], [], y_equilibrium, color=col_mass, alpha=0.1)
time_line, = ax_time.plot([], [], color=col_spring, lw=2)
time_dot, = ax_time.plot([], [], 'o', color=col_mass, markersize=8, zorder=10)

# --- D. Stats Panel ---
ax_stats = fig.add_subplot(gs[:, 2])
ax_stats.set_facecolor(col_panel)
ax_stats.axis('off')
ax_stats.set_xlim(0, 1)
ax_stats.set_ylim(0, 1)

# Title
ax_stats.text(0.5, 0.98, "📊 LIVE DATA", color=col_text, fontsize=11, fontweight='bold',
              ha='center', va='top')

# Parameter display
params_text = f"""
━━━ PARAMETERS ━━━

Mass (m)
  {m} kg

Spring (k)
  {k} N/m

Damping (c)
  {c} Ns/m

Gravity (g)
  {g} m/s²

Rest Length
  {rest_length} m

Equilibrium
  {y_equilibrium:.3f} m
"""
ax_stats.text(0.1, 0.88, params_text, color=col_text_dim, fontsize=9, 
              va='top', ha='left', family='monospace', linespacing=1.3)

# Live values (will be updated)
ax_stats.text(0.1, 0.38, "━━━ LIVE VALUES ━━━", color=col_text, fontsize=9, fontweight='bold')

live_pos = ax_stats.text(0.1, 0.32, "Position: 0.00 m", color=col_mass, fontsize=10, family='monospace')
live_vel = ax_stats.text(0.1, 0.27, "Velocity: 0.00 m/s", color=col_spring, fontsize=10, family='monospace')
live_time = ax_stats.text(0.1, 0.22, "Time: 0.00 s", color=col_text_dim, fontsize=10, family='monospace')

ax_stats.text(0.1, 0.15, "━━━ FORCES (N) ━━━", color=col_text, fontsize=9, fontweight='bold')
live_fg = ax_stats.text(0.1, 0.10, "Gravity: 0.00", color=col_force_g, fontsize=9, family='monospace')
live_fs = ax_stats.text(0.1, 0.06, "Spring: 0.00", color=col_force_s, fontsize=9, family='monospace')
live_fd = ax_stats.text(0.1, 0.02, "Damping: 0.00", color=col_force_d, fontsize=9, family='monospace')

# --- Helper function for realistic spring coil ---
def generate_spring_coil(y_start, y_end, num_coils=12, amplitude=0.15):
    """Generate a realistic zig-zag spring shape."""
    length = y_end - y_start
    if length < 0.1:
        length = 0.1
    
    # Create coil pattern
    n_points = num_coils * 4 + 2
    t = np.linspace(0, 1, n_points)
    
    # Y positions (along spring length)
    sy = y_start + t * length
    
    # X positions (zig-zag pattern)
    sx = np.zeros_like(sy)
    coil_width = amplitude * min(1.0, length / rest_length)  # Compress amplitude when stretched
    
    for i in range(1, len(t) - 1):
        phase = (i - 1) % 4
        if phase == 0:
            sx[i] = coil_width
        elif phase == 1:
            sx[i] = coil_width
        elif phase == 2:
            sx[i] = -coil_width
        else:
            sx[i] = -coil_width
    
    return sx, sy

# --- Animation Update Function ---
def update(frame):
    # Data extraction
    cy = y[frame]
    cv = v[frame]
    t = time_np[frame]

    # Calculate forces
    fg = m * g
    stretch = cy - rest_length
    fs = -k * stretch
    fd = -c * cv
    f_net = fg + fs + fd

    # 1. Update Mass position
    mass_rect.set_y(cy - 0.3)
    mass_glow.set_y(cy - 0.35)
    mass_label.set_position((0, cy))

    # 2. Spring coil visualization
    sx, sy = generate_spring_coil(0, cy - 0.3, num_coils=14, amplitude=0.18)
    spring_line.set_data(sx, sy)
    spring_glow.set_data(sx, sy)

    # 3. Damper
    damper_body.set_y(0.2)
    piston_y = min(cy - 0.3, 0.9)
    damper_piston.set_data([-0.6, -0.6], [0.3, piston_y])
    damper_rod.set_data([-0.6, -0.6], [piston_y, cy - 0.3])

    # 4. Trail
    trail_len = 80
    start = max(0, frame - trail_len)
    trail_x = np.linspace(0.9, 0.9, frame - start)
    trail_line.set_data(trail_x, y[start:frame])

    # 5. Update Force Vectors
    f_scale = 0.012
    mass_center_y = cy
    
    # Gravity (always down)
    fg_len = fg * f_scale
    vec_g.set_positions((0.5, mass_center_y), (0.5, mass_center_y + fg_len))
    lbl_g.set_position((0.55, mass_center_y + fg_len/2))
    lbl_g.set_text(f"Fg={fg:.1f}N")

    # Spring force
    fs_len = fs * f_scale
    if abs(fs_len) > 0.02:
        vec_s.set_positions((-0.5, mass_center_y - 0.15), (-0.5, mass_center_y - 0.15 - fs_len))
        lbl_s.set_position((-0.55, mass_center_y - 0.15 - fs_len/2))
        lbl_s.set_text(f"Fs={fs:.1f}N")
    else:
        vec_s.set_positions((0, 0), (0, 0))
        lbl_s.set_text("")

    # Damping force
    fd_len = fd * f_scale
    if abs(fd_len) > 0.01:
        vec_d.set_positions((0.8, mass_center_y), (0.8, mass_center_y - fd_len))
        lbl_d.set_position((0.85, mass_center_y - fd_len/2))
        lbl_d.set_text(f"Fd={fd:.1f}N")
    else:
        vec_d.set_positions((0, 0), (0, 0))
        lbl_d.set_text("")

    # Net force
    net_len = f_net * f_scale
    if abs(net_len) > 0.02:
        vec_net.set_positions((0, mass_center_y + 0.35), (0, mass_center_y + 0.35 + net_len))
        lbl_net.set_position((0, mass_center_y + 0.35 + net_len + 0.1))
        lbl_net.set_text(f"ΣF={f_net:.1f}N")
    else:
        vec_net.set_positions((0, 0), (0, 0))
        lbl_net.set_text("")

    # 6. Update Phase Plot
    phase_trail.set_data(y[:frame], v[:frame])
    visible_len = min(50, frame)
    phase_line.set_data(y[frame-visible_len:frame], v[frame-visible_len:frame])
    phase_dot.set_data([cy], [cv])
    phase_glow.set_data([cy], [cv])

    # 7. Update Time Plot
    time_line.set_data(time_np[:frame], y[:frame])
    time_dot.set_data([t], [cy])

    # 8. Update Stats Panel
    live_pos.set_text(f"Position: {cy:.3f} m")
    live_vel.set_text(f"Velocity: {cv:.3f} m/s")
    live_time.set_text(f"Time: {t:.2f} s")
    live_fg.set_text(f"Gravity:  {fg:+.2f}")
    live_fs.set_text(f"Spring:   {fs:+.2f}")
    live_fd.set_text(f"Damping:  {fd:+.2f}")

    return (spring_line, spring_glow, mass_rect, mass_glow, mass_label, 
            vec_g, vec_s, vec_d, vec_net, lbl_g, lbl_s, lbl_d, lbl_net,
            phase_trail, phase_line, phase_dot, phase_glow,
            time_line, time_dot, trail_line, 
            damper_piston, damper_rod,
            live_pos, live_vel, live_time, live_fg, live_fs, live_fd)

# Run Animation
print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=steps, interval=25, blit=True)

# Tight layout
plt.tight_layout()

# Save
print("Saving animation (this may take a moment)...")
ani.save('pro_damped_dashboard_jax.gif', writer='pillow', fps=30, dpi=100)
print("✓ Animation saved as 'pro_damped_dashboard_jax.gif'")
# plt.show()
