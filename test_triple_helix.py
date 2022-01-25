import plotly.graph_objects as go
import numpy as np
import pandas as pd

fig = go.Figure()


two_pi = 2 * np.pi
_120 = two_pi * (1 / 3)
_240 = two_pi * (2 / 3)
slider_count = 100
sample_count = 100
t = np.linspace(0, 1, sample_count)

aphase_xdata = np.sin(np.linspace(0, 2 * np.pi, sample_count))
aphase_ydata = np.cos(np.linspace(0, 2 * np.pi, sample_count))
aphase_zdata = np.linspace(0, 1, sample_count)

bphase_xdata = np.sin(np.linspace(0, 2 * np.pi, sample_count) + _120)
bphase_ydata = np.cos(np.linspace(0, 2 * np.pi, sample_count) + _120)
bphase_zdata = np.linspace(0, 1, sample_count)

cphase_xdata = np.sin(np.linspace(0, 2 * np.pi, sample_count) + _240)
cphase_ydata = np.cos(np.linspace(0, 2 * np.pi, sample_count) + _240)
cphase_zdata = np.linspace(0, 1, sample_count)

# for i in range(slider_count):
# aphase_xdata = np.sin(aphase_xdata + (two_pi * (i / slider_count)))
# aphase_ydata = np.sin(aphase_ydata + (two_pi * (i / slider_count)))
# aphase_zdata[i, :] = aphase_zdata

# bphase_xdata = np.sin(bphase_xdata + (two_pi * (i / slider_count))) + _120
# bphase_ydata = np.sin(bphase_ydata + (two_pi * (i / slider_count))) + _120
# bphase_zdata = bphase_zdata

# cphase_xdata = np.sin(cphase_xdata + (two_pi * (i / slider_count))) + _240
# cphase_ydata = np.sin(cphase_ydata + (two_pi * (i / slider_count))) + _240
# cphase_zdata = cphase_zdata

df = pd.DataFrame(
    {
        "aphase_xdata": aphase_xdata,
        "aphase_ydata": aphase_ydata,
        "aphase_zdata": aphase_zdata,
        "bphase_xdata": bphase_xdata,
        "bphase_ydata": bphase_ydata,
        "bphase_zdata": bphase_zdata,
        "cphase_xdata": cphase_xdata,
        "cphase_ydata": cphase_ydata,
        "cphase_zdata": cphase_zdata,
    }
)

#  traces, one for each slider step
# for i, step in enumerate(np.linspace(0, two_pi, slider_count)):
fig.add_trace(
    go.Scatter3d(
        x=aphase_xdata,
        y=aphase_ydata,
        z=aphase_zdata,
        marker={"size": 0},
        line={"color": "black"},
        mode="lines",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=bphase_xdata,
        y=bphase_ydata,
        z=bphase_zdata,
        marker={"size": 0},
        line={"color": "black"},
        mode="lines",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=cphase_xdata,
        y=cphase_ydata,
        z=cphase_zdata,
        marker={"size": 0},
        line={"color": "black"},
        mode="lines",
    )
)


# Make 10th trace visible
fig.data[0].visible = True


# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * len(fig.data)},
            {"title": "Slider switched to step: " + str(i)},
        ],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(active=slider_count, currentvalue={"prefix": "Frequency: "}, pad={"t": 50}, steps=steps)]

fig.update_layout(sliders=sliders)
fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))
# layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))
# fig.update_yaxes(
# scaleanchor="x",
# scaleratio=1,
# )
# fig.update_zaxes(
# scaleanchor="x",
# scaleratio=1,
# )
fig.update_layout(scene_aspectmode="cube")

fig.show()
