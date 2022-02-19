from typing import Any
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, ClientsideFunction
from enum import IntEnum, Enum
import dash_bootstrap_components as dbc
import dash_daq as daq


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


DEBUG = False

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
        {
            "type": "text/javascript",
            "id": "MathJax-callback",
            "src": "assets/mathjax.js",
        },
    ],
)

PHASE_COUNT = 3
AXIS_COUNT = 3
TWO_PI = 2 * np.pi
_120 = TWO_PI * (1 / 3)
_240 = TWO_PI * (2 / 3)
margin = 1
fig = None


class ClarkeParkExploration:
    INSTANCE: "ClarkeParkExploration"

    def __init__(self) -> None:
        self.frequency: float = 1.0
        self.sample_count: int = 100
        self.slider_count: int = 100
        self.three_phase_data: np.ndarray = np.ones((PHASE_COUNT, AXIS_COUNT, self.sample_count))
        self.three_phase_data[:, :] *= np.linspace(0, 1, self.sample_count)
        self.clarke_data: np.ndarray = np.ones((PHASE_COUNT, self.sample_count))
        self.park_data: np.ndarray = np.ones((AXIS_COUNT, self.sample_count))
        self.zeros = np.zeros((self.sample_count))
        self.ones = np.zeros((self.sample_count))
        self.zeros3 = np.zeros((3, self.sample_count))
        self.ones3 = np.zeros((3, self.sample_count))
        self.height = 800
        self.width = self.height * 1.25
        self.projection = ""
        self.projection_label = ""
        self.run_mode = ""
        self.time_offset = 0
        self.frequency = 1
        self.phaseA_offset = 0
        self.phaseB_offset = 0
        self.phaseC_offset = 0
        self.phaseA_amplitude = 0
        self.phaseB_amplitude = 0
        self.phaseC_amplitude = 0
        self.changed_id: Any = None
        self.focus_selection: FocusAxis = FocusAxis.XYZ
        self.time = np.linspace(0, -1, self.sample_count)

        self.first = True

        # Clarke transform
        self.clarke_matrix = (2 / 3) * np.array(
            [
                [1, -(1 / 2), -(1 / 2)],
                [0, (np.sqrt(3) / 2), -(np.sqrt(3) / 2)],
                [(1 / 2), (1 / 2), (1 / 2)],
            ]
        )

        # Park Transform
        self.park_matrix = np.array(
            [
                [
                    self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :],
                    -self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :],
                    self.zeros,
                ],
                [
                    self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :],
                    self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :],
                    self.zeros,
                ],
                [
                    self.zeros,
                    self.zeros,
                    self.ones,
                ],
            ]
        )

        ClarkeParkExploration.INSTANCE = self

    def generate_three_phase_data(self) -> np.ndarray:
        self.time_plus_offset = self.time + self.time_offset

        self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :] = self.phaseA_amplitude * np.cos(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseA_offset * np.pi)
        )
        self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :] = self.phaseA_amplitude * np.sin(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseA_offset * np.pi)
        )

        self.three_phase_data[PhaseEnum.B, AxisEnum.Y, :] = self.phaseB_amplitude * np.cos(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseB_offset * np.pi) - _120
        )
        self.three_phase_data[PhaseEnum.B, AxisEnum.Z, :] = self.phaseB_amplitude * np.sin(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseB_offset * np.pi) - _120
        )

        self.three_phase_data[PhaseEnum.C, AxisEnum.Y, :] = self.phaseC_amplitude * np.cos(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseC_offset * np.pi) + _120
        )
        self.three_phase_data[PhaseEnum.C, AxisEnum.Z, :] = self.phaseC_amplitude * np.sin(
            self.frequency * TWO_PI * self.time_plus_offset + (self.phaseC_offset * np.pi) + _120
        )

    def do_clarke_transform(self):
        """Perform Clarke transform function.

        https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
        https://www.mathworks.com/help/physmod/sps/ref/clarketransform.html
        """
        # Clarke transform function
        self.clarke_data[:, :] = np.dot(
            self.clarke_matrix,
            np.array(
                [
                    self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :],
                    self.three_phase_data[PhaseEnum.B, AxisEnum.Y, :],
                    self.three_phase_data[PhaseEnum.C, AxisEnum.Y, :],
                ]
            ),
        )

    def do_park_transform(self):
        """Perform Park transform function.

        https://de.wikipedia.org/wiki/D/q-Transformation
        https://www.mathworks.com/help/physmod/sps/ref/clarketoparkangletransform.html
        """

        # create Park transformation matrix, with reference based on enum value
        self.park_matrix[0, 0, :] = self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :]
        self.park_matrix[1, 0, :] = -self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :]
        self.park_matrix[0, 1, :] = self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :]
        self.park_matrix[1, 1, :] = self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :]

        # perform the matrix math
        self.park_data = np.einsum(
            "ijk,ik->jk",
            self.park_matrix,
            self.clarke_data,
        )

    def generate_figure_data(self):
        self.generate_three_phase_data()
        self.do_clarke_transform()
        self.do_park_transform()
        self.figure_data = {
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": self.three_phase_data[PhaseEnum.A, AxisEnum.Y, :],
                    "z": self.three_phase_data[PhaseEnum.A, AxisEnum.Z, :],
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
                    "x": self.three_phase_data[PhaseEnum.B, AxisEnum.X, :],
                    "y": self.three_phase_data[PhaseEnum.B, AxisEnum.Y, :],
                    "z": self.three_phase_data[PhaseEnum.B, AxisEnum.Z, :],
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
                    "x": self.three_phase_data[PhaseEnum.C, AxisEnum.X, :],
                    "y": self.three_phase_data[PhaseEnum.C, AxisEnum.Y, :],
                    "z": self.three_phase_data[PhaseEnum.C, AxisEnum.Z, :],
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
                        self.three_phase_data[PhaseEnum.A, AxisEnum.Y, 0],
                    ],
                    "z": [
                        0,
                        self.three_phase_data[PhaseEnum.A, AxisEnum.Z, 0],
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
                        self.three_phase_data[PhaseEnum.B, AxisEnum.Y, 0],
                    ],
                    "z": [
                        0,
                        self.three_phase_data[PhaseEnum.B, AxisEnum.Z, 0],
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
                        self.three_phase_data[PhaseEnum.C, AxisEnum.Y, 0],
                    ],
                    "z": [
                        0,
                        self.three_phase_data[PhaseEnum.C, AxisEnum.Z, 0],
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": self.clarke_data[ClarkeEnum.A, :],
                    "z": self.zeros,
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": self.zeros,
                    "z": self.clarke_data[ClarkeEnum.B, :],
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": self.clarke_data[ClarkeEnum.Z, :],
                    "z": self.clarke_data[ClarkeEnum.Z, :],
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
                    "y": [0, self.clarke_data[ClarkeEnum.A, 0]],
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
                    "z": [0, self.clarke_data[ClarkeEnum.B, 0]],
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
                    "y": [0, self.clarke_data[ClarkeEnum.Z, 0]],
                    "z": [0, self.clarke_data[ClarkeEnum.Z, 0]],
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": np.cos(
                        self.frequency * TWO_PI * self.time_plus_offset[0]
                        + (self.phaseA_offset * np.pi)
                        + (np.pi / 2)
                    )
                    * self.park_data[ParkEnum.D, :],
                    "z": np.sin(
                        self.frequency * TWO_PI * self.time_plus_offset[0]
                        + (self.phaseA_offset * np.pi)
                        + (np.pi / 2)
                    )
                    * self.park_data[ParkEnum.D, :],
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
                    "x": self.three_phase_data[PhaseEnum.A, AxisEnum.X, :],
                    "y": np.cos(
                        self.frequency * TWO_PI * self.time_plus_offset[0] + (self.phaseA_offset * np.pi)
                    )
                    * self.park_data[ParkEnum.Q, :],
                    "z": np.sin(
                        self.frequency * TWO_PI * self.time_plus_offset[0] + (self.phaseA_offset * np.pi)
                    )
                    * self.park_data[ParkEnum.Q, :],
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
                        np.cos(
                            self.frequency * TWO_PI * self.time_plus_offset[0]
                            + (self.phaseA_offset * np.pi)
                            + (np.pi / 2)
                        )
                        * self.park_data[ParkEnum.D, 0],
                    ],
                    "z": [
                        0,
                        np.sin(
                            self.frequency * TWO_PI * self.time_plus_offset[0]
                            + (self.phaseA_offset * np.pi)
                            + (np.pi / 2)
                        )
                        * self.park_data[ParkEnum.D, 0],
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
                        np.cos(
                            self.frequency * TWO_PI * self.time_plus_offset[0] + (self.phaseA_offset * np.pi)
                        )
                        * self.park_data[ParkEnum.Q, 0],
                    ],
                    "z": [
                        0,
                        np.sin(
                            self.frequency * TWO_PI * self.time_plus_offset[0] + (self.phaseA_offset * np.pi)
                        )
                        * self.park_data[ParkEnum.Q, 0],
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
        if self.first is False:
            self.figure_data["layout"]["uirevision"] = 1
            self.figure_data["layout"]["scene"]["aspectratio"] = {
                "x": 1,
                "y": 1,
                "z": 1,
            }
            self.figure_data["layout"]["height"] = self.height
            self.figure_data["layout"]["width"] = self.width
            self.figure_data["layout"]["margin"] = {
                "l": margin,
                "r": margin,
                "t": margin,
                "b": margin,
            }

        else:
            self.first = False
            self.figure_data["layout"]["uirevision"] = 1
            self.figure_data["layout"]["height"] = self.height
            self.figure_data["layout"]["width"] = self.width
            self.figure_data["layout"]["scene_aspectmode"] = "cube"
            self.figure_data["layout"]["autosize"] = False
            self.figure_data["layout"]["scene"]["aspectmode"] = "manual"
            self.figure_data["layout"]["scene"]["aspectratio"] = {
                "x": 1,
                "y": 1,
                "z": 1,
            }
            self.figure_data["layout"]["margin"] = {
                "l": margin,
                "r": margin,
                "t": margin,
                "b": margin,
            }

        if self.focus_selection == FocusAxis.XY:
            self.figure_data["layout"]["scene"]["camera"] = {
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
        elif self.focus_selection == FocusAxis.XZ:
            self.figure_data["layout"]["scene"]["camera"] = {
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
        elif self.focus_selection == FocusAxis.YZ:
            self.figure_data["layout"]["scene"]["camera"] = {
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
        elif self.focus_selection == FocusAxis.XYZ:
            self.figure_data["layout"]["scene"]["camera"] = {
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
        self.figure_data["layout"]["scene"]["camera"]["projection"] = {
            "type": self.projection,
        }

    @staticmethod
    @app.callback(
        [
            Output("scatter_plot", "figure"),
            Output("three_phase_data", "children"),
            Output("clarke_data", "children"),
            Output("park_data", "children"),
            Output("projection", "label"),
            Output("run-mode", "label"),
            Output("interval-component", "max_intervals"),
            Output("time_slider", "value"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("time_slider", "value"),
            Input("frequency_slider", "value"),
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
            Input("projection", "on"),
            Input("run-mode", "on"),
        ],
    )
    def update_graphs(
        interval,
        time_slider,
        frequency_slider,
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
        projection_isometric,
        run_mode,
    ):
        self = ClarkeParkExploration.INSTANCE
        self.time_offset = time_slider
        self.frequency = frequency_slider
        self.phaseA_offset = phaseA_phase_slider
        self.phaseB_offset = phaseB_phase_slider
        self.phaseC_offset = phaseC_phase_slider
        self.phaseA_amplitude = phaseA_amplitude_slider
        self.phaseB_amplitude = phaseB_amplitude_slider
        self.phaseC_amplitude = phaseC_amplitude_slider
        self.height = size_slider
        self.width = size_slider * 1.25
        if projection_isometric is False:
            self.projection = "isometric"
            self.projection_label = "Enable Orthographic Projection"
        else:
            self.projection = "orthographic"
            self.projection_label = "Disable Orthographic Projection"
        if run_mode is True:
            self.run_mode = "Disable Continuous Mode"
            max_intervals = 10000000
            self.time_offset += 1.0 / self.slider_count
            if self.time_offset > 1:
                self.time_offset = 0
        else:
            self.run_mode = "Enable Continuous Mode"
            max_intervals = 0
        self.changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
        if "focus_xy" in self.changed_id:
            self.focus_selection = FocusAxis.XY
        elif "focus_xz" in self.changed_id:
            self.focus_selection = FocusAxis.XZ
        elif "focus_yz" in self.changed_id:
            self.focus_selection = FocusAxis.YZ
        elif "focus_corner" in self.changed_id:
            self.focus_selection = FocusAxis.XYZ
        self.generate_figure_data()
        return [
            self.figure_data,
            html.Td(
                [
                    html.Tr(
                        [
                            html.Td(f"{self.three_phase_data[PhaseEnum.A, AxisEnum.X, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.A, AxisEnum.Y, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.A, AxisEnum.Z, 0]:0.2f}\u00A0\u00A0"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.three_phase_data[PhaseEnum.B, AxisEnum.X, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.B, AxisEnum.Y, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.B, AxisEnum.Z, 0]:0.2f}\u00A0\u00A0"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.three_phase_data[PhaseEnum.C, AxisEnum.X, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.C, AxisEnum.Y, 0]:0.2f}\u00A0\u00A0"),
                            html.Td(f"{self.three_phase_data[PhaseEnum.C, AxisEnum.Z, 0]:0.2f}\u00A0\u00A0"),
                        ]
                    ),
                ]
            ),
            html.Td(
                [
                    html.Tr(
                        [
                            html.Td(f"{self.clarke_data[ClarkeEnum.A, 0]:0.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.clarke_data[ClarkeEnum.B, 0]:0.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.clarke_data[ClarkeEnum.Z, 0]:0.2f}"),
                        ]
                    ),
                ]
            ),
            html.Td(
                [
                    html.Tr(
                        [
                            html.Td(f"{self.park_data[ParkEnum.D, 0]:0.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.park_data[ParkEnum.Q, 0]:0.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"{self.park_data[ParkEnum.Z, 0]:0.2f}"),
                        ]
                    ),
                ]
            ),
            self.projection_label,
            self.run_mode,
            max_intervals,
            self.time_offset,
        ]


cpe = ClarkeParkExploration()
app.layout = dbc.Container(
    [
        html.H1("Interactive Clarke & Park Transforms"),
        html.Div(
            [
                html.H3("Overview"),
                html.P(
                    "This interactive plot is meant to be used for exploring The interactions of "
                    + "variables in a three-phase system and the Clarke and Park transforms."
                ),
                html.H3("Introduction"),
                html.P(
                    "The Plotted lines can be turned on or off by clicking on the plot legend.  "
                    + "The buttons below can be used to set the view to various fixed perspectives.  "
                    + "The graph can be zoomed and panned by interacting with it using the mouse and menu.  "
                    + "The sliders under the graph can be used to adjust variables used in the graph.  "
                    + "The math below will update along with the graph to help better understant effects "
                    + "that the sliders have on the graph"
                ),
            ]
        ),
        html.H3("Equations"),
        html.H5(
            "If the math below does not render, try refreshing and/or force refreshing by pressing CTRL+F5.",
            style={"color": "darkorange"},
        ),
        html.P("Three phase helix data."),
        html.Table(
            html.Tr(
                [
                    html.Td(
                        "$$ \\begin{bmatrix}  A_x(t) & A_y(t) & A_z(t) \\\\ B_x(t) & B_y(t) & B_z(t) \\\\ C_x(t) & C_y(t) & C_z(t)  \\end{bmatrix} = \\begin{bmatrix}  x(t) & sin(t) & cos(t) \\\\ x(t) & sin(t+\\frac{2*\\pi}{3})) & cos(t+\\frac{2*\\pi}{3}) \\\\ x(t) & sin(t-\\frac{2*\\pi}{3}) & cos(t-\\frac{2*\\pi}{3})  \\end{bmatrix} $$"
                    ),
                    html.Td("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"),
                    html.Td(
                        [
                            html.P(
                                "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 For t = time slider below."
                            ),
                            html.P(
                                "$$ \\begin{bmatrix}  A_x(slider) & A_y(slider) & A_z(slider) \\\\ B_x(slider) & B_y(slider) & B_z(slider) \\\\ C_x(slider) & C_y(slider) & C_z(slider) \\end{bmatrix} = $$"
                            ),
                        ]
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
                        "$$ \\frac{2}{3} \\begin{bmatrix} 1 & -\\frac{1}{2} & -\\frac{1}{2} \\\\ 0 & \\frac{\\sqrt{3}}{2} & -\\frac{\\sqrt{3}}{2} \\\\ \\frac{1}{2} & \\frac{1}{2} & \\frac{1}{2} \\end{bmatrix}\\begin{bmatrix} A_z(t) \\\\ B_z(t) \\\\ C_z(t) \\end{bmatrix} = \\begin{bmatrix}  \\alpha(t) \\\\ \\beta(t)  \\\\ Z_{C}(t) \\end{bmatrix} $$"
                    ),
                    html.Td("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"),
                    html.Td(
                        [
                            html.P(
                                "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 For t = time slider below."
                            ),
                            html.P(
                                "$$ \\begin{bmatrix}  \\alpha(slider) \\\\ \\beta(slider)  \\\\ Z_{C}(slider) \\end{bmatrix} = $$",
                            ),
                        ]
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
                        "$$ \\begin{bmatrix} sin(\\omega t) & -cos(\\omega t) & 0 \\\\ cos(\\omega t) & sin(\\omega t) & 0 \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} \\alpha(t) \\\\ \\beta(t) \\\\ Z_{C}(t) \\end{bmatrix} = \\begin{bmatrix} d(t) \\\\ q(t) \\\\ Z_{P}(t) \\end{bmatrix} $$"
                    ),
                    html.Td("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"),
                    html.Td(
                        [
                            html.P(
                                "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 For t = time slider below."
                            ),
                            html.P(
                                "$$ \\begin{bmatrix} d(slider) \\\\ q(slider) \\\\ Z_{P}(slider) \\end{bmatrix} = $$"
                            ),
                        ]
                    ),
                    html.Td(id="park_data"),
                ]
            )
        ),
        html.H3("View"),
        html.Div(
            [
                html.P("Use the following controls to change the perspective of the graph."),
                html.P("Sometimes you have to switch views more than once for plotly to reset any rotation."),
            ]
        ),
        html.Table(
            html.Tr(
                [
                    html.Td(
                        daq.BooleanSwitch(
                            id="projection", on=False, label="Isometric Projection", labelPosition="top"
                        ),
                    ),
                    html.Td(
                        html.P("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"),
                    ),
                    html.Td(
                        daq.BooleanSwitch(
                            id="run-mode", on=False, label="Enable Continuous Mode", labelPosition="top"
                        ),
                    ),
                ]
            )
        ),
        html.P(),
        html.Div(
            [
                html.Button("View X/Y (real / cosine)", id="focus_xy", n_clicks=0),
                html.Button("View X/Z (imaginary / sine)", id="focus_xz", n_clicks=0),
                html.Button("View Y/Z (polar)", id="focus_yz", n_clicks=0),
                html.Button("View X/Y/Z", id="focus_corner", n_clicks=0),
            ]
        ),
        html.P(),
        html.H3("Graph"),
        dcc.Graph(
            id="scatter_plot",
            style={
                "scene_aspectmode": "cube",
            },
        ),
        html.H3("Controls"),
        html.P(
            [
                html.P("Use this slider to adjust the time axis by adding an " + "offset from zero to one."),
                dcc.Slider(
                    id="time_slider",
                    min=0,
                    max=1,
                    marks={0: "0", 1: "1"},
                    step=1 / cpe.slider_count,
                    value=0,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P("This slider adjusts the frequency of the sine waves (\\(  \\omega \\) )."),
                dcc.Slider(
                    id="frequency_slider",
                    min=0.5,
                    max=5,
                    marks={0.5: "0.5", 5: "5"},
                    step=1 / cpe.slider_count,
                    value=1,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Table(
                    html.Tr(
                        [
                            html.Td(
                                [
                                    html.P("This slider controls the amplitude of Phase A."),
                                    dcc.Slider(
                                        id="phaseA_amplitude_slider",
                                        min=0.1,
                                        max=2,
                                        marks={0.1: "0.1", 2: "2"},
                                        step=1 / cpe.slider_count,
                                        value=1,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                            html.Td(
                                [
                                    html.P("This slider controls the amplitude of Phase B."),
                                    dcc.Slider(
                                        id="phaseB_amplitude_slider",
                                        min=0.1,
                                        max=2,
                                        marks={0.1: "0.1", 2: "2"},
                                        step=1 / cpe.slider_count,
                                        value=1,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                            html.Td(
                                [
                                    html.P("This slider controls the amplitude of Phase C."),
                                    dcc.Slider(
                                        id="phaseC_amplitude_slider",
                                        min=0.1,
                                        max=2,
                                        marks={0.1: "0.1", 2: "2"},
                                        step=1 / cpe.slider_count,
                                        value=1,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                    style={"width": "100%"},
                ),
                html.Table(
                    html.Tr(
                        [
                            html.Td(
                                [
                                    html.P("This slider adds a phase offset to Phase A."),
                                    dcc.Slider(
                                        id="phaseA_phase_slider",
                                        min=-1,
                                        max=1,
                                        marks={-1: "-1", 1: "1"},
                                        step=1 / cpe.slider_count,
                                        value=0,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                            html.Td(
                                [
                                    html.P("This slider adds a phase offset to Phase B."),
                                    dcc.Slider(
                                        id="phaseB_phase_slider",
                                        min=-1,
                                        max=1,
                                        marks={-1: "-1", 1: "1"},
                                        step=1 / cpe.slider_count,
                                        value=0,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                            html.Td(
                                [
                                    html.P("This slider adds a phase offset to Phase C."),
                                    dcc.Slider(
                                        id="phaseC_phase_slider",
                                        min=-1,
                                        max=1,
                                        marks={-1: "-1", 1: "1"},
                                        step=1 / cpe.slider_count,
                                        value=0,
                                        updatemode="drag",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ]
                            ),
                        ]
                    ),
                    style={"width": "100%"},
                ),
                html.P("This slider adjusts the size of the graphic."),
                dcc.Slider(
                    id="size_slider",
                    min=400,
                    max=1600,
                    marks={400: "400", 1600: "1600"},
                    step=cpe.slider_count,
                    value=700,
                    updatemode="drag",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.P(id="ignore"),
                dcc.Interval(id="interval-component", interval=250, n_intervals=0, max_intervals=0),
            ]
        ),
    ],
)
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="mathjax_call"),
    Output("ignore", "children"),
    Input("time_slider", "value"),
    Input("frequency_slider", "value"),
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
    Input("projection", "on"),
)


if DEBUG is True:
    app.run_server(debug=True)
else:
    app.run_server("0.0.0.0", 8050)
