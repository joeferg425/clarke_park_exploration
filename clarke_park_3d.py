from matplotlib.pyplot import cla
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from enum import IntEnum


class FocusAxis(IntEnum):
    XY = 0
    XZ = 1
    YZ = 2
    XYZ = 3
    NONE = 4


focus_selection = FocusAxis.XYZ

app = dash.Dash(__name__)

two_pi = 2 * np.pi
_120 = two_pi * (1 / 3)
_240 = two_pi * (2 / 3)
slider_count = 100
sample_count = 100

data = np.ones((3, 3, sample_count))
data[:, :] *= np.linspace(0, 1, sample_count)
data[:, [1, 2]] *= 2 * np.pi

radian_offset = 0
fig = None


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="scatter-plot",
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
        html.Div(
            [
                html.Button("Focus X/Y", id="focus-xy", n_clicks=0),
                html.Button("Focus X/Z", id="focus-xz", n_clicks=0),
                html.Button("Focus Y/Z", id="focus-yz", n_clicks=0),
                html.Button("Focus X/Y/Z", id="focus-corner", n_clicks=0),
            ]
        ),
    ],
)


def generate_figure_data():
    global data, radian_offset, focus_selection
    figure_data = {
        "data": [
            {
                "x": data[0, 0],
                "y": np.sin(data[0, 1] + (radian_offset * 2 * np.pi)),
                "z": np.cos(data[0, 2] + (radian_offset * 2 * np.pi)),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase A",
            },
            {
                "x": data[1, 0],
                "y": np.sin(data[1, 1] + (radian_offset * 2 * np.pi) + _120),
                "z": np.cos(data[1, 2] + (radian_offset * 2 * np.pi) + _120),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase B",
            },
            {
                "x": data[2, 0],
                "y": np.sin(data[2, 1] + (radian_offset * 2 * np.pi) + _240),
                "z": np.cos(data[2, 2] + (radian_offset * 2 * np.pi) + _240),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase C",
            },
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
    if focus_selection == FocusAxis.XYZ:
        figure_data["layout"]["scene"]["camera"] = {
            "up": {
                "x": 0.0,
                "y": 0.5,
                "z": 0.0,
            },
            "eye": {
                "x": 1.75,
                "y": 1.75,
                "z": 1.75,
            },
        }
    elif focus_selection == FocusAxis.XY:
        figure_data["layout"]["scene"]["camera"] = {
            "up": {
                "x": 0.0,
                "y": 0.5,
                "z": 0.0,
            },
            "eye": {
                "x": 0.0,
                "y": 0.0,
                "z": 2.0,
            },
        }
    elif focus_selection == FocusAxis.XZ:
        figure_data["layout"]["scene"]["camera"] = {
            "up": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.5,
            },
            "eye": {
                "x": 0.0,
                "y": 2.0,
                "z": 0.0,
            },
        }
    elif focus_selection == FocusAxis.YZ:
        figure_data["layout"]["scene"]["camera"] = {
            "up": {
                "x": 0.0,
                "y": 0.5,
                "z": 0.0,
            },
            "eye": {
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
            },
        }
    # else:
    #     figure_data["layout"]["scene"]["camera"] = {
    #         "eye": graph.figure.layout.scene.eye,
    #         "up": {
    #             "x": 0.0,
    #             "y": 0.0,
    #             "z": 0.0,
    #         },
    #     }

    focus_selection = FocusAxis.NONE

    return figure_data


@app.callback(
    Output("scatter-plot", "figure"),
    [
        Input("range-slider", "value"),
        Input("focus-xy", "n_clicks"),
        Input("focus-xz", "n_clicks"),
        Input("focus-yz", "n_clicks"),
        Input("focus-corner", "n_clicks"),
    ],
)
def update_graphs(slider_range, btn1, btn2, btn3, btn4):
    global radian_offset, focus_selection
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if "focus-xy" in changed_id:
        focus_selection = FocusAxis.XY
    elif "focus-xz" in changed_id:
        focus_selection = FocusAxis.XZ
    elif "focus-yz" in changed_id:
        focus_selection = FocusAxis.YZ
    elif "focus-corner" in changed_id:
        focus_selection = FocusAxis.XYZ
    radian_offset = slider_range
    return generate_figure_data()


app.run_server(debug=True)
