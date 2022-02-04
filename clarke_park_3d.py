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
    Z = 2


class ParkEnum(IntEnum):
    D = 0
    Q = 1
    Z = 2


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
    ClarkeZ = "#22CC22"
    ParkD = "#F0E442"
    ParkQ = "#999999"


class DashEnum(Enum):
    Normal = "solid"
    Clarke = "dot"
    Park = "dash"


class WidthEnum(Enum):
    Time = 3
    Phasor = 7
    Clarke = 9
    Park = 7


class FocusAxis(IntEnum):
    XY = 0
    XZ = 1
    YZ = 2
    XYZ = 3
    NONE = 4


focus_selection = FocusAxis.XYZ
app = dash.Dash(
    __name__,
    title="Clarke & Park Transforms",
    external_stylesheets=[dbc.themes.DARKLY],
    external_scripts=[
        {
            "type": "text/javascript",
            "id": "MathJax-script",
            "src": "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML",
        },
    ],
)

two_pi = 2 * np.pi
_120 = two_pi * (1 / 3)
_240 = two_pi * (2 / 3)
slider_count = 100
sample_count = 100
first = True
height = 800
width = height * 1.25
margin = 1

data = np.ones((3, 3, sample_count))
data[:, :] *= np.linspace(0, 1, sample_count)
data[:, [AxisEnum.Y, AxisEnum.Z]] *= 2 * np.pi
clarke = None
park = None
zeros = np.zeros((sample_count))
ones = np.zeros((sample_count))
zeros3 = np.zeros((3, sample_count))
ones3 = np.zeros((3, sample_count))
time_offset = 0
phaseA_offset = 0
phaseB_offset = 0
phaseC_offset = 0
phaseA_amplitude = 0
phaseB_amplitude = 0
phaseC_amplitude = 0
fig = None


def do_clarke_transform():
    """Perform Clarke transform function.

    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
    https://www.mathworks.com/help/physmod/sps/ref/clarketransform.html
    """
    global data, clarke
    # Clarke transform
    clarke_matrix = (2 / 3) * np.array(
        [
            [1, -(1 / 2), -(1 / 2)],
            [0, (np.sqrt(3) / 2), -(np.sqrt(3) / 2)],
            [(1 / 2), (1 / 2), (1 / 2)],
        ]
    )
    # Clarke transform function
    clarke = np.dot(
        clarke_matrix,
        np.array(
            [
                np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                np.sin(phaseB_offset + data[PhaseEnum.B, AxisEnum.Y, :] + (time_offset * 2 * np.pi) + _120),
                np.sin(phaseC_offset + data[PhaseEnum.C, AxisEnum.Y, :] + (time_offset * 2 * np.pi) + _240),
            ]
        )
        * np.array([phaseA_amplitude, phaseB_amplitude, phaseC_amplitude])[:, None],
    )
    return clarke


def do_park_transform():
    """Perform Park transform function.

    https://de.wikipedia.org/wiki/D/q-Transformation
    https://www.mathworks.com/help/physmod/sps/ref/clarketoparkangletransform.html
    """
    global data, clarke, park
    # create Park transformation matrix, with reference based on enum value
    park_matrix = np.array(
        [
            [
                np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                -np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                zeros,
            ],
            [
                np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                zeros,
            ],
            [
                zeros,
                zeros,
                ones,
            ],
        ]
    )
    # perform the matrix math
    park = np.einsum(
        "ijk,ik->jk",
        park_matrix,
        clarke,
    )
    return park


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
        html.H2("If the math below does not render, press CTRL+F5."),
        html.P("Three phase helix data."),
        html.Table(
            html.Tr(
                [
                    html.Td(
                        "$$ \\begin{bmatrix}  A_x & A_y & A_z \\\\ B_x & B_y & B_z \\\\ C_x & C_y & C_z  \\end{bmatrix} = \\begin{bmatrix}  x(t) & sin(t) & cos(t) \\\\ x(t) & sin(t+\\frac{2*\\pi}{3})) & cos(t+\\frac{2*\\pi}{3}) \\\\ x(t) & sin(t-\\frac{2*\\pi}{3}) & cos(t-\\frac{2*\\pi}{3})  \\end{bmatrix} = $$"
                    ),
                    html.Td(id="three_phase_data"),
                ]
            )
        ),
        html.P("Clarke transform data."),
        html.Table(
            html.Tr(
                [
                    html.Td(
                        "$$ \\frac{2}{3} \\begin{bmatrix} 1 & -\\frac{1}{2} & -\\frac{1}{2} \\\\ 0 & \\frac{\\sqrt{3}}{2} & -\\frac{\\sqrt{3}}{2} \\\\ \\frac{1}{2} & \\frac{1}{2} & \\frac{1}{2} \\end{bmatrix}\\begin{bmatrix} A_y \\\\ B_y \\\\ C_y \\end{bmatrix} = \\begin{bmatrix}  \\alpha \\\\ \\beta   \\\\ Z_{C} \\end{bmatrix} = $$"
                    ),
                    html.Td(id="clarke_data"),
                ]
            )
        ),
        html.P("Park Transform data."),
        html.Table(
            html.Tr(
                [
                    html.Td(
                        "$$ \\begin{bmatrix} sin(\\Theta) & -cos(\\Theta) & 0 \\\\ cos(\\Theta) & sin(\\Theta) & 0 \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} \\alpha \\\\ \\beta \\\\ Z_{C} \\end{bmatrix} = \\begin{bmatrix} d \\\\ q \\\\ Z_{P} \\end{bmatrix} = $$"
                    ),
                    html.Td(id="park_data"),
                ]
            )
        ),
        html.Div(
            [
                dcc.Graph(
                    id="scatter_plot",
                    style={
                        "scene_aspectmode": "cube",
                    },
                )
            ],
        ),
        html.Div(
            [
                html.P("Sometimes you have to switch views more than once for plotly to reset any rotation."),
                html.Button("View X/Y (real / sine)", id="focus_xy", n_clicks=0),
                html.Button("View X/Z (imaginary / cosine)", id="focus_xz", n_clicks=0),
                html.Button("View Y/Z (polar)", id="focus_yz", n_clicks=0),
                html.Button("View X/Y/Z", id="focus_corner", n_clicks=0),
            ]
        ),
        html.Div(
            [
                html.P(
                    "Use this slider to adjust the time axis by adding an "
                    + "offset from zero to one (the signal is 1 Hertz)."
                ),
                dcc.Slider(
                    id="time_slider",
                    min=0,
                    max=1,
                    step=1 / slider_count,
                    value=0,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider controls Phase A amplitude."),
                dcc.Slider(
                    id="phaseA_amplitude_slider",
                    min=0.1,
                    max=2,
                    step=1 / slider_count,
                    value=1,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider controls Phase B amplitude."),
                dcc.Slider(
                    id="phaseB_amplitude_slider",
                    min=0.1,
                    max=2,
                    step=1 / slider_count,
                    value=1,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider controls Phase C amplitude."),
                dcc.Slider(
                    id="phaseC_amplitude_slider",
                    min=0.1,
                    max=2,
                    step=1 / slider_count,
                    value=1,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider adds a phase offset to Phase A."),
                dcc.Slider(
                    id="phaseA_phase_slider",
                    min=-np.pi,
                    max=np.pi,
                    step=1 / slider_count,
                    value=0,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider adds a phase offset to Phase B."),
                dcc.Slider(
                    id="phaseB_phase_slider",
                    min=-np.pi,
                    max=np.pi,
                    step=1 / slider_count,
                    value=0,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider adds a phase offset to Phase C."),
                dcc.Slider(
                    id="phaseC_phase_slider",
                    min=-np.pi,
                    max=np.pi,
                    step=1 / slider_count,
                    value=0,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider adjusts the size of the graphic."),
                dcc.Slider(
                    id="size_slider",
                    min=400,
                    max=1600,
                    step=100,
                    value=700,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ]
        ),
    ],
)


def generate_figure_data():
    global data, time_offset, focus_selection, clarke, park, first
    do_clarke_transform()
    do_park_transform()
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
                "y": phaseA_amplitude
                * np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, :] + (time_offset * 2 * np.pi)),
                "z": phaseA_amplitude
                * np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, :] + (time_offset * 2 * np.pi)),
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
                "y": phaseB_amplitude
                * np.sin(phaseB_offset + data[PhaseEnum.B, AxisEnum.Y, :] + (time_offset * 2 * np.pi) + _120),
                "z": phaseB_amplitude
                * np.cos(phaseB_offset + data[PhaseEnum.B, AxisEnum.Z, :] + (time_offset * 2 * np.pi) + _120),
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
                "y": phaseC_amplitude
                * np.sin(phaseC_offset + data[PhaseEnum.C, AxisEnum.Y, :] + (time_offset * 2 * np.pi) + _240),
                "z": phaseC_amplitude
                * np.cos(phaseC_offset + data[PhaseEnum.C, AxisEnum.Z, :] + (time_offset * 2 * np.pi) + _240),
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
                "y": [
                    0,
                    phaseA_amplitude
                    * np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, 0] + (time_offset * 2 * np.pi)),
                ],
                "z": [
                    0,
                    phaseA_amplitude
                    * np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, 0] + (time_offset * 2 * np.pi)),
                ],
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
                "y": [
                    0,
                    phaseB_amplitude
                    * np.sin(
                        phaseB_offset + data[PhaseEnum.B, AxisEnum.Y, 0] + (time_offset * 2 * np.pi) + _120
                    ),
                ],
                "z": [
                    0,
                    phaseB_amplitude
                    * np.cos(
                        phaseB_offset + data[PhaseEnum.B, AxisEnum.Z, 0] + (time_offset * 2 * np.pi) + _120
                    ),
                ],
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
                "y": [
                    0,
                    phaseC_amplitude
                    * np.sin(
                        phaseC_offset + data[PhaseEnum.C, AxisEnum.Y, 0] + (time_offset * 2 * np.pi) + _240
                    ),
                ],
                "z": [
                    0,
                    phaseC_amplitude
                    * np.cos(
                        phaseC_offset + data[PhaseEnum.C, AxisEnum.Z, 0] + (time_offset * 2 * np.pi) + _240
                    ),
                ],
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
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": clarke[ClarkeEnum.A, :],
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
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": zeros,
                "z": clarke[ClarkeEnum.B, :],
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
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": clarke[ClarkeEnum.Z, :],
                "z": clarke[ClarkeEnum.Z, :],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke Zero (t)",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeZ.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, clarke[ClarkeEnum.A, 0]],
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
                "z": [0, clarke[ClarkeEnum.B, 0]],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke β",
                "line": {
                    "width": WidthEnum.Clarke.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeB.value,
                },
            },
            {
                "x": [0, 0],
                "y": [0, clarke[ClarkeEnum.Z, 0]],
                "z": [0, clarke[ClarkeEnum.Z, 0]],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Clarke Zero ",
                "line": {
                    "width": WidthEnum.Time.value,
                    "dash": DashEnum.Clarke.value,
                    "color": ColorEnum.ClarkeZ.value,
                },
            },
            {
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, 0] + (time_offset * 2 * np.pi))
                * park[ParkEnum.D, :],
                "z": np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, 0] + (time_offset * 2 * np.pi))
                * park[ParkEnum.D, :],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Park d (t)",
                "line": {
                    "width": WidthEnum.Park.value,
                    "dash": DashEnum.Park.value,
                    "color": ColorEnum.ParkD.value,
                },
            },
            {
                "x": data[PhaseEnum.A, AxisEnum.X, :],
                "y": np.sin(
                    phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, 0] + (time_offset * 2 * np.pi) + (np.pi / 2)
                )
                * park[ParkEnum.Q, :],
                "z": np.cos(
                    phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, 0] + (time_offset * 2 * np.pi) + (np.pi / 2)
                )
                * park[ParkEnum.Q, :],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Park q (t)",
                "line": {
                    "width": WidthEnum.Park.value,
                    "dash": DashEnum.Park.value,
                    "color": ColorEnum.ParkQ.value,
                },
            },
            {
                "x": [0, 0],
                "y": [
                    0,
                    np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, 0] + (time_offset * 2 * np.pi))
                    * park[ParkEnum.D, 0],
                ],
                "z": [
                    0,
                    np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, 0] + (time_offset * 2 * np.pi))
                    * park[ParkEnum.D, 0],
                ],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Park d",
                "line": {
                    "width": WidthEnum.Park.value,
                    "dash": DashEnum.Park.value,
                    "color": ColorEnum.ParkD.value,
                },
            },
            {
                "x": [0, 0],
                "y": [
                    0,
                    np.sin(
                        phaseA_offset
                        + data[PhaseEnum.A, AxisEnum.Y, 0]
                        + (time_offset * 2 * np.pi)
                        + (np.pi / 2)
                    )
                    * park[ParkEnum.Q, 0],
                ],
                "z": [
                    0,
                    np.cos(
                        phaseA_offset
                        + data[PhaseEnum.A, AxisEnum.Z, 0]
                        + (time_offset * 2 * np.pi)
                        + (np.pi / 2)
                    )
                    * park[ParkEnum.Q, 0],
                ],
                "type": "scatter3d",
                "mode": "lines",
                "name": "Park q",
                "line": {
                    "width": WidthEnum.Park.value,
                    "dash": DashEnum.Park.value,
                    "color": ColorEnum.ParkQ.value,
                },
            },
        ],
        "layout": {
            "scene": {
                "xaxis": {
                    "title": "x (Time)",
                    "tickvals": [-1, 0, 1],
                },
                "yaxis": {
                    "title": "y (Real)",
                    "tickvals": [-1, 0, 1],
                },
                "zaxis": {
                    "title": "z (Imaginary)",
                    "tickvals": [-1, 0, 1],
                },
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
        figure_data["layout"]["height"] = height
        figure_data["layout"]["width"] = width
        figure_data["layout"]["margin"] = {
            "l": margin,
            "r": margin,
            "t": margin,
            "b": margin,
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
        figure_data["layout"]["margin"] = {
            "l": margin,
            "r": margin,
            "t": margin,
            "b": margin,
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
    # Output("scatter_plot", "figure"),
    [
        Output("scatter_plot", "figure"),
        Output("three_phase_data", "children"),
        Output("clarke_data", "children"),
        Output("park_data", "children"),
    ],
    [
        Input("time_slider", "value"),
        Input("phaseA_amplitude_slider", "value"),
        Input("phaseB_amplitude_slider", "value"),
        Input("phaseC_amplitude_slider", "value"),
        Input("phaseA_phase_slider", "value"),
        Input("phaseB_phase_slider", "value"),
        Input("phaseC_phase_slider", "value"),
        Input("size_slider", "value"),
        Input("focus_xy", "n_clicks"),
        Input("focus_xz", "n_clicks"),
        Input("focus_yz", "n_clicks"),
        Input("focus_corner", "n_clicks"),
    ],
)
def update_graphs(
    time_slider,
    phaseA_amplitude_slider,
    phaseB_amplitude_slider,
    phaseC_amplitude_slider,
    phaseA_phase_slider,
    phaseB_phase_slider,
    phaseC_phase_slider,
    size_slider,
    btn1,
    btn2,
    btn3,
    btn4,
):
    global time_offset, phaseA_amplitude, phaseB_amplitude, phaseC_amplitude, phaseA_offset, phaseB_offset, phaseC_offset, focus_selection, height, width, clarke, park
    time_offset = time_slider
    phaseA_offset = phaseA_phase_slider
    phaseB_offset = phaseB_phase_slider
    phaseC_offset = phaseC_phase_slider
    phaseA_amplitude = phaseA_amplitude_slider
    phaseB_amplitude = phaseB_amplitude_slider
    phaseC_amplitude = phaseC_amplitude_slider
    height = size_slider
    width = size_slider * 1.25
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if "focus_xy" in changed_id:
        focus_selection = FocusAxis.XY
    elif "focus_xz" in changed_id:
        focus_selection = FocusAxis.XZ
    elif "focus_yz" in changed_id:
        focus_selection = FocusAxis.YZ
    elif "focus_corner" in changed_id:
        focus_selection = FocusAxis.XYZ
    return [
        generate_figure_data(),
        html.Td(
            [
                html.Tr(
                    [
                        html.Td(
                            f"{phaseA_offset + data[PhaseEnum.A, AxisEnum.X, 0] + (time_offset * 2 * np.pi):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseA_amplitude * np.sin(phaseA_offset + data[PhaseEnum.A, AxisEnum.Y, 0] + (time_offset * 2 * np.pi)):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseA_amplitude * np.cos(phaseA_offset + data[PhaseEnum.A, AxisEnum.Z, 0] + (time_offset * 2 * np.pi)):0.2f}\u00A0\u00A0"
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            f"{phaseB_offset + data[PhaseEnum.B, AxisEnum.X, 0] + (time_offset * 2 * np.pi):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseB_amplitude * np.sin(phaseB_offset + data[PhaseEnum.B, AxisEnum.Y, 0] + (time_offset * 2 * np.pi) + _120):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseB_amplitude * np.cos(phaseB_offset + data[PhaseEnum.B, AxisEnum.Z, 0] + (time_offset * 2 * np.pi) + _120):0.2f}\u00A0\u00A0"
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            f"{phaseC_offset + data[PhaseEnum.C, AxisEnum.X, 0] + (time_offset * 2 * np.pi):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseC_amplitude * np.sin(phaseC_offset + data[PhaseEnum.C, AxisEnum.Y, 0] + (time_offset * 2 * np.pi) + _240):0.2f}\u00A0\u00A0"
                        ),
                        html.Td(
                            f"{phaseC_amplitude * np.cos(phaseC_offset + data[PhaseEnum.C, AxisEnum.Z, 0] + (time_offset * 2 * np.pi) + _240):0.2f}\u00A0\u00A0"
                        ),
                    ]
                ),
            ]
        ),
        html.Td(
            [
                html.Tr(
                    [
                        html.Td(f"{clarke[ClarkeEnum.A, 0]:0.2f}"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(f"{clarke[ClarkeEnum.B, 0]:0.2f}"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(f"{clarke[ClarkeEnum.Z, 0]:0.2f}"),
                    ]
                ),
            ]
        ),
        html.Td(
            [
                html.Tr(
                    [
                        html.Td(f"{park[ParkEnum.D, 0]:0.2f}"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(f"{park[ParkEnum.Q, 0]:0.2f}"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(f"{park[ParkEnum.Z, 0]:0.2f}"),
                    ]
                ),
            ]
        ),
    ]


app.run_server(debug=True)
