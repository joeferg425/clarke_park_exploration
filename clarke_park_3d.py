from matplotlib.pyplot import cla
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from enum import IntEnum

from sqlalchemy import false


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
first = True

data = np.ones((3, 3, sample_count))
data[:, :] *= np.linspace(0, 1, sample_count)
data[:, [1, 2]] *= 2 * np.pi
zeros = np.zeros((sample_count))
radian_offset = 0
fig = None


def do_clarke_transform():
    """Perform Clarke transform function.

    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
    https://www.mathworks.com/help/physmod/sps/ref/clarketransform.html
    """
    global data
    # Clarke transform
    clarke_matrix = (2 / 3) * np.array(
        [
            [1, -(1 / 2), -(1 / 2)],
            [0, (np.sqrt(3) / 2), -(np.sqrt(3) / 2)],
            [(1 / 2), (1 / 2), (1 / 2)],
        ]
    )
    # Clarke transform function
    clarke_alpha_beta_zero_array = np.dot(
        clarke_matrix,
        np.array(
            [
                np.sin(data[0, 1, :]),
                np.sin(data[1, 1, :] + (radian_offset * 2 * np.pi) + _120),
                np.sin(data[2, 1, :] + (radian_offset * 2 * np.pi) + _240),
            ]
        )
        # * np.array([self.amplitude1, self.amplitude2, self.amplitude3])[:, None],
    )
    # assign outputs to individual variables for polar plots
    # (
    #     self.clarke_alpha_instantaneous,
    #     self.clarke_beta_instantaneous,
    #     self.clarke_zero_instantaneous,
    # ) = self.clarke_alpha_beta_zero_array[:, 0]
    # # matplotlib likes positive vector lengths
    # if self.clarke_alpha_instantaneous >= 0:
    #     self.clarke_alpha_instantaneous_angle = 0.0
    # else:
    #     self.clarke_alpha_instantaneous_angle = np.pi
    #     self.clarke_alpha_instantaneous *= -1
    # if self.clarke_beta_instantaneous >= 0:
    #     self.clarke_beta_instantaneous_angle = np.pi * 3 / 2
    # else:
    #     self.clarke_beta_instantaneous_angle = np.pi / 2
    #     self.clarke_beta_instantaneous *= -1
    return clarke_alpha_beta_zero_array


# def do_park_transform(self):
#     """Perform Park transform function.

#     https://de.wikipedia.org/wiki/D/q-Transformation
#     https://www.mathworks.com/help/physmod/sps/ref/clarketoparkangletransform.html
#     """
#     # create Park transformation matrix, with reference based on enum value
#     if self.park_alignment == ParkAlignment.q_aligned:
#         self.park_matrix = np.array(
#             [
#                 [
#                     np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     -np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.zeros((self.sample_count)),
#                 ],
#                 [
#                     np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.zeros((self.sample_count)),
#                 ],
#                 [
#                     np.zeros((self.sample_count)),
#                     np.zeros((self.sample_count)),
#                     np.ones((self.sample_count)),
#                 ],
#             ]
#         )
#     elif self.park_alignment == ParkAlignment.d_aligned:
#         self.park_matrix = np.array(
#             [
#                 [
#                     np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.zeros((self.sample_count)),
#                 ],
#                 [
#                     np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     -np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
#                     np.zeros((self.sample_count)),
#                 ],
#                 [
#                     np.zeros((self.sample_count)),
#                     np.zeros((self.sample_count)),
#                     np.ones((self.sample_count)),
#                 ],
#             ]
#         )
#     # perform the matrix math
#     self.park_array = np.einsum(
#         "ijk,ik->jk",
#         self.park_matrix,
#         self.clarke_alpha_beta_zero_array,
#     )
#     # save instantaneous values for polar plots
#     (
#         self.park_d_instantaneous,
#         self.park_q_instantaneous,
#         self.park_zero_instantaneous,
#     ) = self.park_array[:, 0]
#     # set the phase angle based on reference value
#     if self.park_alignment == ParkAlignment.q_aligned:
#         self.park_q_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0]
#         self.park_d_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0] - np.pi / 2
#     else:
#         self.park_d_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0]
#         self.park_q_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0] + np.pi / 2
#     # matplotlib likes positive vectors
#     if self.park_d_instantaneous < 0:
#         self.park_d_instantaneous *= -1
#         self.park_d_instantaneous_angle += 2 * np.pi
#     if self.park_q_instantaneous < 0:
#         self.park_q_instantaneous *= -1
#         self.park_q_instantaneous_angle += 2 * np.pi


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
    global data, radian_offset, focus_selection, first
    clarke = do_clarke_transform()
    figure_data = {
        "data": [
            {
                "x": data[0, 0, :],
                "y": np.sin(data[0, 1, :] + (radian_offset * 2 * np.pi)),
                "z": np.cos(data[0, 2, :] + (radian_offset * 2 * np.pi)),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase A",
            },
            {
                "x": data[1, 0, :],
                "y": np.sin(data[1, 1, :] + (radian_offset * 2 * np.pi) + _120),
                "z": np.cos(data[1, 2, :] + (radian_offset * 2 * np.pi) + _120),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase B",
            },
            {
                "x": data[2, 0, :],
                "y": np.sin(data[2, 1, :] + (radian_offset * 2 * np.pi) + _240),
                "z": np.cos(data[2, 2, :] + (radian_offset * 2 * np.pi) + _240),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase C",
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[0, 1, 0] + (radian_offset * 2 * np.pi))],
                "z": [0, np.cos(data[0, 2, 0] + (radian_offset * 2 * np.pi))],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor A",
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[1, 1, 0] + (radian_offset * 2 * np.pi) + _120)],
                "z": [0, np.cos(data[1, 2, 0] + (radian_offset * 2 * np.pi) + _120)],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor B",
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[2, 1, 0] + (radian_offset * 2 * np.pi) + _240)],
                "z": [0, np.cos(data[2, 2, 0] + (radian_offset * 2 * np.pi) + _240)],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor C",
            },
            {
                "x": data[0, 0, :],
                "y": clarke[0, :],
                "z": zeros,
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke α(t)",
            },
            {
                "x": data[0, 0, :],
                "y": zeros,
                "z": clarke[1, :],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke β(t)",
            },
            {
                "x": [0, 0],
                "y": [0, clarke[0, 0]],
                "z": [0, 0],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke α",
            },
            {
                "x": [0, 0],
                "y": [0, 0],
                "z": [0, clarke[1, 0]],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke β",
            },
        ],
    }
    if first is False:
        figure_data["layout"] = {
            "uirevision": 1,
            "xaxis": {"range": [-1, 1]},
            "yaxis": {"range": [-1, 1]},
            "zaxis": {"range": [-1, 1]},
            "scene": {
                # "aspectmode": "manual",
                "aspectratio": {
                    "x": 1,
                    "y": 1,
                    "z": 1,
                },
            },
        }
    else:
        first = False
        figure_data["layout"] = {
            "uirevision": 1,
            "height": 700,
            "width": 700,
            "xaxis": {"range": [-1, 1]},
            "yaxis": {"range": [-1, 1]},
            "zaxis": {"range": [-1, 1]},
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

    # focus_selection = FocusAxis.NONE

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
