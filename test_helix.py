import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from sqlalchemy import true

app = dash.Dash(__name__)

slider_count = 100
sample_count = 100
t = np.linspace(0, 1, sample_count)
x = np.sin(2 * np.pi * t + (np.pi * (0)))
y = np.cos(2 * np.pi * t + (np.pi * (0)))


fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        # df[mask],
        x=x,
        y=y,
        z=t,
        # marker={"size": 0},
        # line={"color": "black"},
        mode="lines",
        # aspectmode="manual",
        # aspectratio=[1, 0.2, 0.2],
    )
)
fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    # scene_aspectmode="cube",
    # aspectratio=[1, 1, 1],
    # aspectmode="manual",
)
fig.update_layout(
    scene_aspectmode="manual",
    scene_aspectratio=dict(x=1, y=1, z=1),
)

# fig.plotly_relayout(
# {"scene_aspectmode": "cube"},
# )

# t = np.linspace(0, 1, sample_count)
# for i, step in enumerate(np.linspace(0, two_pi, slider_count)):
#     fig.add_trace(
#         go.Scatter3d(
#             visible=False,
#             x=np.sin(2 * np.pi * t + (two_pi * (i / slider_count))),
#             y=np.cos(2 * np.pi * t + (two_pi * (i / slider_count))),
#             z=t,
#             marker={"size": 0},
#             line={"color": "black"},
#             mode="lines",
#         )
#     )


# Make first trace visible
# fig.data[0].visible = True


# Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[
#             {"visible": [False] * len(fig.data)},
#             {"title": "Slider switched to step: " + str(i)},
#         ],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(active=slider_count, currentvalue={"prefix": "Frequency: "}, pad={"t": 50}, steps=steps)]

# fig.update_layout(sliders=sliders)
# fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))
app.layout = html.Div(
    [
        # html.H6("Change the value in the text box to see callbacks in action!"),
        # html.Br(),
        html.Div(
            [
                dcc.Graph(
                    id="scatter-plot",
                    figure=fig,
                    style={
                        "height": 700,
                        "width": 700,
                        "scene_aspectmode": "cube",
                    },
                )
            ],
            style={
                "height": "700v",
                "width": "700v",
                "scene_aspectmode": "cube",
            },
        ),
        # html.Div([dcc.Graph(id="scatter-plot")], style={"height": 700, "width": 700}),
        # html.Br(),
        html.Div(["Input: ", dcc.Input(id="my-input", value="initial value", type="text")]),
        html.Div(
            [
                dcc.Slider(
                    id="range-slider",
                    min=0,
                    max=1,
                    step=1 / slider_count,
                    value=0,
                    updatemode="drag",
                )
            ]
        ),
        # html.Br(),
        # html.Div(id="my-output"),
    ],
    # style={"height": "100vh"},
    # scene={"scene_aspectmode": "cube"},
)

# fig.update_layout(
#     width=800,
#     height=700,
#     autosize=False,
#     scene_aspectmode="cube",
#     # aspectratio=dict(x=1, y=1, z=1),
#     # aspectmode="manual",
# )


# @app.callback(
#     Output(component_id="my-output", component_property="children"),
#     Input(component_id="my-input", component_property="value"),
# )
# def update_output_div(input_value):
#     return "Output: {}".format(input_value)


@app.callback(Output("scatter-plot", "figure"), [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    print(slider_range)
    # low, high = slider_range
    # mask = (df.petal_width > low) & (df.petal_width < high)
    x = np.sin(2 * np.pi * t + (np.pi * (slider_range)))
    y = np.cos(2 * np.pi * t + (np.pi * (slider_range)))

    # fig = px.scatter_3d(
    # fig = go.Scatter3d(
    # {"x": x, "y": y, "z": t},
    # df[mask],
    # x=x,
    # y=y,
    # z=t,
    # marker={"size": 0},
    # line={"color": "black"},
    # mode="lines",
    # style={
    # height=700,
    # width=700,
    # mode="lines",
    # "scene_aspectmode": "cube",
    # },
    # )
    # return fig
    # d = fig.
    # return d
    return {
        "data": [
            {
                "x": x,
                "y": y,
                "z": t,
                "type": "scatter3d",
                "mode": "lines",
            }
        ],
        "layout": {
            "height": 700,
            "width": 700,
            "scene_aspectmode": "cube",
            "autosize": False,
            "scene": {
                "aspectmode": "manual",
                "aspectratio": {
                    "x": 1,
                    "y": 1,
                    "z": 1,
                },
            },
        },
    }


# app.


# fig.show()
app.run_server(debug=True)
