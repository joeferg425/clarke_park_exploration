import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

two_pi = 2 * np.pi
_120 = two_pi * (1 / 3)
_240 = two_pi * (2 / 3)
slider_count = 100
sample_count = 100

data = np.ones((3, 3, sample_count))
data[:, :] *= np.linspace(0, 1, sample_count)
data[:, [1, 2]] *= 2 * np.pi


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="scatter-plot",
                    # figure=fig,
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
    ],
)


@app.callback(Output("scatter-plot", "figure"), [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    return {
        "data": [
            {
                "x": data[0, 0],
                "y": np.sin(data[0, 1] + (slider_range * 2 * np.pi)),
                "z": np.cos(data[0, 2] + (slider_range * 2 * np.pi)),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase A",
            },
            {
                "x": data[1, 0],
                "y": np.sin(data[1, 1] + (slider_range * 2 * np.pi) + _120),
                "z": np.cos(data[1, 2] + (slider_range * 2 * np.pi) + _120),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase B",
            },
            {
                "x": data[2, 0],
                "y": np.sin(data[2, 1] + (slider_range * 2 * np.pi) + _240),
                "z": np.cos(data[2, 2] + (slider_range * 2 * np.pi) + _240),
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


app.run_server(debug=True)
