# pac_line_anim.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

# Canvas
W, H = 16, 9
BG = "#0b1a2b"
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Pac setup
r = 1.0
px = 1.5
py = H * 0.5
speed = 0.08
mouth_max = np.deg2rad(50)

face = patches.Circle((px, py), r, facecolor="yellow", edgecolor="none", zorder=5)
eye = patches.Circle((px + 0.3 * r, py + 0.45 * r), 0.1, facecolor=BG, zorder=6)
ax.add_patch(face)
ax.add_patch(eye)

def mouth_patch(angle_rad):
    v1 = (px, py)
    v2 = (px + r * np.cos(angle_rad), py + r * np.sin(angle_rad))
    v3 = (px + r * np.cos(-angle_rad), py + r * np.sin(-angle_rad))
    return patches.Polygon([v1, v2, v3], closed=True, facecolor=BG, edgecolor=BG, zorder=6)

mouth = mouth_patch(0.01)
ax.add_patch(mouth)

# File icons
def file_icon(cx, cy, s=0.8):
    body = patches.FancyBboxPatch((cx - 0.45 * s, cy - 0.6 * s), 0.9 * s, 1.2 * s,
                                  boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor="#e9ecef", edgecolor="#c9ced6", linewidth=1, zorder=3)
    fold = patches.Polygon([[cx + 0.25 * s, cy + 0.6 * s],
                            [cx + 0.45 * s, cy + 0.4 * s],
                            [cx + 0.45 * s, cy + 0.6 * s]],
                            closed=True, facecolor="#d6dbe3", edgecolor="#c9ced6", zorder=4)
    return [body, fold]

N = 6
spacing = 1.8
start_x = 5
y_line = py
icons = []
for i in range(N):
    cx = start_x + i * spacing
    grp = file_icon(cx, y_line, s=0.9)
    for p in grp:
        ax.add_patch(p)
    icons.append(grp)

eaten = np.zeros(N, dtype=bool)

# Animation
def update(frame):
    global px, mouth
    px += speed
    if px - r > W + 0.5:
        px = 1.5
        eaten[:] = False
        for grp in icons:
            for p in grp:
                p.set_alpha(1.0)

    phase = 2 * np.pi * (frame % 60) / 60.0
    a = 0.1 + (mouth_max - 0.1) * (0.5 * (1 + np.sin(phase)))

    face.center = (px, py)
    eye.center = (px + 0.3 * r, py + 0.45 * r)

    # remove and redraw mouth properly
    mouth.remove()
    new_mouth = mouth_patch(a)
    ax.add_patch(new_mouth)
    mouth = new_mouth

    for i, grp in enumerate(icons):
        if eaten[i]:
            continue
        bbox = grp[0].get_bbox()
        cx = bbox.x0 + bbox.width / 2
        cy = bbox.y0 + bbox.height / 2
        dist = np.hypot(cx - px, cy - py)
        ang = abs(np.arctan2(cy - py, cx - px))
        if dist < 0.9 * r and ang < a:
            eaten[i] = True
            for p in grp:
                p.set_alpha(0.0)

    return []

ani = animation.FuncAnimation(fig, update, frames=600, interval=16, blit=False)

try:
    from matplotlib.animation import PillowWriter
    ani.save("pac_line_anim.gif", writer=PillowWriter(fps=60))
    print("Saved: pac_line_anim.gif")
except Exception:
    plt.show()
