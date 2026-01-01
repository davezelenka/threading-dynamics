import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


def count_vertices_spatial(traj, eps=0.08, min_samples=25):
    """
    Count stable vertices as spatial recurrence clusters in 3D.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(traj)
    labels = clustering.labels_

    # Exclude noise points
    unique_labels = set(labels)
    unique_labels.discard(-1)

    return max(len(unique_labels), 1)


# -----------------------------
# Ω₃ Map Definition
# -----------------------------
def omega3_map(state, a=2.5, d=0.6, L=1.2, b=0.0):
    x, y, z = state
    amp = lambda u: a * np.tanh(u)
    fold = lambda u: np.sign(u) * (np.clip(np.abs(u), 0, L) ** 2)

    x_next = d * (amp(y) - fold(x)) + b * np.sin(z)
    y_next = d * (amp(z) - fold(y)) + b * np.sin(x)
    z_next = d * (amp(x) - fold(z)) + b * np.sin(y)

    return np.array([x_next, y_next, z_next])

# -----------------------------
# Setup Figure
# -----------------------------
v=1
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('#0a0a14')
ax.set_facecolor('#0a0a14')

ax.set_xlabel('X', color='white', fontsize=10, weight='bold')
ax.set_ylabel('Y', color='white', fontsize=10, weight='bold')
ax.set_zlabel('Z', color='white', fontsize=10, weight='bold')

ax.tick_params(colors='white', labelsize=9)
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#1a1a2e')
ax.grid(True, alpha=0.2, color='white')

# Set fixed axis limits
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-2.5, 2.5])

# Title
title_text = ax.text2D(0.5, 0.98, 'Ω₃ SYMMETRY BREAKING', transform=ax.transAxes,
                       fontsize=14, color='white', ha='center', weight='bold')

# Info text
info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color='white', 
                      fontsize=11, family='monospace', verticalalignment='top')

# -----------------------------
# Animation Settings
# -----------------------------
b_min = 0.0 #0
b_max = 1 #0.15
num_frames = 3000  # Total frames for full b sweep 600
max_trail_segments = 100  # Number of line segments to keep 200
steps_per_frame = 1  # Evolution steps per frame

# Create line objects for trail
trail_lines = []
for _ in range(max_trail_segments):
    line, = ax.plot([], [], [], linewidth=1.5)
    trail_lines.append(line)

# Current point
point, = ax.plot([0], [0], [0], 'o', color='#ff3366', markersize=8, 
                 markeredgecolor='white', markeredgewidth=1.5)

# Trail history
trail_history = []

# Current state
current_state = np.array([0.01, 0.02, -0.015])

# Vertex buffer for counting
vertex_buffer = []
vertex_buffer_max = 5000

# Pre-compute vertex counts at key b values for reference
print("Pre-computing vertex counts...")
vertex_reference = {}
for b_test in np.linspace(b_min, b_max, 100):
    # --- Vertex counting ---
    state = np.array([0.01, 0.02, -0.015])
    for _ in range(3000):
        state = omega3_map(state, b=b_test)

    traj = []
    for _ in range(5000):
        state = omega3_map(state, b=b_test)
        traj.append(state.copy())

    traj = np.array(traj)
    verts = count_vertices_spatial(traj)

    vertex_reference[round(b_test, 3)] = verts
    print(f"  b={b_test:.3f} → {verts} vertices")
print("Done!\n")


# Determine phase
def get_phase2(v1, v2):
    if v1 == v2:
        return "ORDERED"
    else:
        return "CHAOTIC"

# Get color for phase
def get_phase_color(v1, v2):
    if v1 == v2:
        return "cyan"
    else:
        return "yellow"

# Get line width for phase
def get_line_width(b_val):
    
        return 1.0

# -----------------------------
# Animation Function
# -----------------------------
def animate(frame):
    global current_state, trail_history, vertex_buffer, v
    
    # Reset at start
    if frame == 0:
        current_state = np.array([0.01, 0.02, -0.015])
        trail_history = [current_state.copy()]
        vertex_buffer = []
        
        # Burn-in
        for _ in range(500):
            current_state = omega3_map(current_state, b=b_min)
            vertex_buffer.append(current_state.copy())
    
    # Current b value
    b_val = b_min + (b_max - b_min) * (frame / num_frames)
    
    # Evolve system
    for _ in range(steps_per_frame):
        current_state = omega3_map(current_state, b=b_val)
        trail_history.append(current_state.copy())
        vertex_buffer.append(current_state.copy())
    
    # Keep trail history limited
    if len(trail_history) > max_trail_segments + 1:
        trail_history = trail_history[-(max_trail_segments + 1):]
    
    # Keep vertex buffer limited
    if len(vertex_buffer) > vertex_buffer_max:
        vertex_buffer = vertex_buffer[-vertex_buffer_max:]
    
    # Update trail lines with fading alpha (1.0 to 0.0)
    # Get vertex count
    # --- Vertex counting ---
    state = np.array([0.01, 0.02, -0.015])
    for _ in range(3000):
        state = omega3_map(state, b=b_val)

    traj = []
    for _ in range(5000):
        state = omega3_map(state, b=b_val)
        traj.append(state.copy())

    traj = np.array(traj)

    verts = count_vertices_spatial(traj)



    num_segments = len(trail_history) - 1
    phase_color = get_phase_color(v, verts)
    phase_width = get_line_width(b_val)
    
    for i in range(max_trail_segments):
        if i < num_segments:
            # Get segment (newest first)
            idx = num_segments - i - 1
            p1 = trail_history[idx]
            p2 = trail_history[idx + 1]
            
            # Update line
            trail_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            trail_lines[i].set_3d_properties([p1[2], p2[2]])
            
            # Calculate alpha: newest = 1.0, oldest = 0.0 (LINEAR fade)
            alpha = 1.0 - (i / max_trail_segments)
            trail_lines[i].set_alpha(max(0, alpha))
            trail_lines[i].set_visible(True)
            trail_lines[i].set_color(phase_color)
            trail_lines[i].set_linewidth(phase_width)
        else:
            # Hide unused lines
            trail_lines[i].set_visible(False)
    
    # Update current point
    point.set_data([current_state[0]], [current_state[1]])
    point.set_3d_properties([current_state[2]])
    
    



    phase = get_phase2(v, verts)
    v = verts
    
    # Update info text
    info_str = f"""b = {b_val:.4f}
Phase: {phase}
Vertices: {v}"""
    
    info_text.set_text(info_str)
    
    return trail_lines + [point, info_text]

# Create animation
print(f"Starting animation...")
print(f"b range: {b_min:.4f} → {b_max:.4f}")
print(f"Total frames: {num_frames}")
print(f"Trail segments: {max_trail_segments}")
print("="*70)

anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()

# Optional: Save
# print("\nSaving animation...")
# anim.save('omega3_animation.mp4', writer='ffmpeg', fps=20, dpi=120)
# print("Done!")