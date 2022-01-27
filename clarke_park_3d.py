import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from enum import IntEnum, Enum
import dash_bootstrap_components as dbc


class AxisEnum(IntEnum):
    X = 0
    Y = 1
    Z = 2


class PhaseEnum(IntEnum):
    A = 0
    B = 1
    C = 2


class ClarkeEnum(IntEnum):
    A = 0
    B = 1


class ColorEnum(Enum):
    """This is supposed to be a colorblind-friendly palette.

    Args:
        Enum: color enum
    """

    PhaseA = "#E69F00"
    PhaseB = "#0072B2"
    PhaseC = "#CC79A7"
    ClarkeA = "#D55E00"
    ClarkeB = "#56B4E9"
    Other1 = "#F0E442"
    Other2 = "#999999"


class DashEnum(Enum):
    Normal = "solid"
    Clarke = "dash"


class WidthEnum(Enum):
    Time = 2
    Phasor = 5
    Clarke = 6


class FocusAxis(IntEnum):
    XY = 0
    XZ = 1
    YZ = 2
    XYZ = 3
    NONE = 4


focus_selection = FocusAxis.XYZ

app = dash.Dash(__name__, title="Clarke & Park Transforms", external_stylesheets=[dbc.themes.DARKLY])

two_pi = 2 * np.pi
_120 = two_pi * (1 / 3)
_240 = two_pi * (2 / 3)
slider_count = 100
sample_count = 100
first = True
height = 1000
width = 1200

data = np.ones((3, 3, sample_count))
data[:, :] *= np.linspace(0, 1, sample_count)
data[:, [AxisEnum.Y, AxisEnum.Z]] *= 2 * np.pi
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
                np.sin(data[PhaseEnum.A, AxisEnum.Y, :] + (radian_offset * 2 * np.pi)),
                np.sin(data[PhaseEnum.B, AxisEnum.Y, :] + (radian_offset * 2 * np.pi) + _120),
                np.sin(data[PhaseEnum.B, AxisEnum.Y, :] + (radian_offset * 2 * np.pi) + _240),
            ]
        )
        # * np.array([self.amplitude1, self.amplitude2, self.amplitude3])[:, None],
    )
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


# app.layout = dbc.Div(
app.layout = dbc.Container(
    [
        html.H1("Interactive Clarke & Park Transforms"),
        html.Div(
            [
                html.P("Plotted lines can be turned on or off by clicking on the plot legend."),
                html.P("Buttons near bottom of screen can be used to set the veiw to fixed perspectives."),
                html.P("Graph can be zoomed and panned by interacting with it using the mouse and menu."),
                html.P("Sliders near bottom of screen can be used to adjust variables used in the graph."),
            ]
        ),
        html.Div(
            [
                dcc.Graph(
                    id="scatter-plot",
                    style={
                        "height": height,
                        "width": width,
                        "scene_aspectmode": "cube",
                    },
                )
            ],
            style={
                "height": height,
                "width": width,
                "scene_aspectmode": "cube",
            },
        ),
        html.Div(
            [
                html.P(
                    "Use this slider to adjust the time axis by adding an offset from zero to one (the signal is 1 Hertz)."
                ),
            ]
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
                    tooltip={"placement": "bottom", "always_visible": True},
                )
            ]
        ),
        html.Div(
            [
                html.Button("Focus X/Y (real / sine)", id="focus-xy", n_clicks=0),
                html.Button("Focus X/Z (imaginary / cosine)", id="focus-xz", n_clicks=0),
                html.Button("Focus Y/Z (polar)", id="focus-yz", n_clicks=0),
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
                "x": [0, 1],
                "y": [-1, 1],
                "z": [-1, 1],
                "type": "scatter3d",
                "mode": "lines",
                "name": "fixed_xyz_range",
                "line": {
                    "width": 0,
                    "color": "rgba(0,0,0,0)",
                },
            },
            {
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": np.sin(data[PhaseEnum.A, AxisEnum.Y, :] + (radian_offset * 2 * np.pi)),
                "z": np.cos(data[PhaseEnum.A, AxisEnum.Z, :] + (radian_offset * 2 * np.pi)),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase A (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseA.value,
                },
            },
            {
                "x": data[PhaseEnum.B, AxisEnum.X, :],
                "y": np.sin(data[PhaseEnum.B, AxisEnum.Y, :] + (radian_offset * 2 * np.pi) + _120),
                "z": np.cos(data[PhaseEnum.B, AxisEnum.Z, :] + (radian_offset * 2 * np.pi) + _120),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase B (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseB.value,
                },
            },
            {
                "x": data[PhaseEnum.C, AxisEnum.X, :],
                "y": np.sin(data[PhaseEnum.C, AxisEnum.Y, :] + (radian_offset * 2 * np.pi) + _240),
                "z": np.cos(data[PhaseEnum.C, AxisEnum.Z, :] + (radian_offset * 2 * np.pi) + _240),
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phase C (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseC.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[0, 1, 0] + (radian_offset * 2 * np.pi))],
                "z": [0, np.cos(data[0, 2, 0] + (radian_offset * 2 * np.pi))],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor A",
                "line": {
                    "width": WidthEnum.Phasor.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseA.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[1, 1, 0] + (radian_offset * 2 * np.pi) + _120)],
                "z": [0, np.cos(data[1, 2, 0] + (radian_offset * 2 * np.pi) + _120)],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor B",
                "line": {
                    "width": WidthEnum.Phasor.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseB.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, np.sin(data[2, 1, 0] + (radian_offset * 2 * np.pi) + _240)],
                "z": [0, np.cos(data[2, 2, 0] + (radian_offset * 2 * np.pi) + _240)],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Phasor C",
                "line": {
                    "width": WidthEnum.Phasor.value,
                    "dash": DashEnum.Normal.value,
                    "color": ColorEnum.PhaseC.value,
                },
            },
            {
                "x": data[0, 0, :],
                "y": clarke[0, :],
                "z": zeros,
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke α (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeA.value,
                },
            },
            {
                "x": data[0, 0, :],
                "y": zeros,
                "z": clarke[1, :],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke β (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeB.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, clarke[0, 0]],
                "z": [0, 0],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke α",
                "line": {
                    "width": WidthEnum.Clarke.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeA.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, 0],
                "z": [0, clarke[1, 0]],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke β",
                "line": {
                    "width": WidthEnum.Clarke.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeB.value,
                },
            },
        ],
        "layout": {
            "scene": {
                "xaxis": {"title": "x (Time)"},
                "yaxis": {"title": "y (Real)"},
                "zaxis": {"title": "z (Imaginary)"},
            },
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        },
    }
    if first is False:
        figure_data["layout"]["uirevision"] = 1
        figure_data["layout"]["scene"]["aspectratio"] = {
            "x": 1,
            "y": 1,
            "z": 1,
        }

    else:
        first = False
        figure_data["layout"]["uirevision"] = 1
        figure_data["layout"]["height"] = height
        figure_data["layout"]["width"] = width
        figure_data["layout"]["scene_aspectmode"] = "cube"
        figure_data["layout"]["autosize"] = False
        figure_data["layout"]["scene"]["aspectmode"] = "manual"
        figure_data["layout"]["scene"]["aspectratio"] = {
            "x": 1,
            "y": 1,
            "z": 1,
        }

    if focus_selection == FocusAxis.XY:
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
                "y": -2.0,
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
                "x": -2.0,
                "y": 0.0,
                "z": 0.0,
            },
        }
    elif focus_selection == FocusAxis.XYZ:
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
