"""This file is meant to be used as an educational and exploratory tool.

Please modify it as needed to better understand Clarke & Park transforms.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from enum import IntEnum


class ParkAlignment(IntEnum):
    """This enumerates the two park transform reference options.

    Args:
        IntEnum: The enumeration value
    """

    q_aligned = 1
    d_aligned = 2


class PhaseIndex(IntEnum):
    """Enumerates indices in three phase data matrix.

    Args:
        IntEnum: enumeration value
    """

    Phase1 = 0
    Phase2 = 1
    Phase3 = 2


class ClarkeIndex(IntEnum):
    """Enumerates indices in three clarke transform data matrix.

    Args:
        IntEnum: enumeration value
    """

    Alpha = 0
    Beta = 1
    Zero = 2


class ParkIndex(IntEnum):
    """Enumerates indices in park transform data matrix.

    Args:
        IntEnum: enumeration value
    """

    D = 0
    Q = 1
    Zero = 2


class ClarkeParkDemo:
    """Container class for doing Clarke and Park Transforms."""

    def __init__(self) -> None:
        """This is a container class meant to allow cleaner calculation code."""
        # configuration constants
        self.frequency1 = 1
        self.frequency2 = 1
        self.frequency3 = 1
        self.amplitude1 = 1
        self.amplitude2 = 1
        self.amplitude3 = 1
        self.duration = 1
        self.sample_rate = 100
        self.sample_count = self.duration * self.sample_rate
        self.park_alignment = ParkAlignment.d_aligned

        # calculated constants
        self._2pi = 2 * np.pi
        self._120 = (1 / 3) * self._2pi
        self.phase_offset_1 = self._120 * (0)
        self.phase_offset_2 = self._120 * (1)
        self.phase_offset_3 = self._120 * (2)

        # run functions
        self.do_three_phase_time_domain_data()
        self.do_clarke_transform()
        self.do_park_transform()

        # plot data
        self.pyplot_initial_plot()

    def do_three_phase_time_domain_data(self):
        """Create three phases of sinusoidal time-domain data."""
        # time array
        self.time_array = np.linspace(0, self.duration, self.sample_count)
        # create three sine waves0
        self.three_phase_data_matrix = np.array(
            [
                self.frequency1 * 2 * np.pi * self.time_array + self.phase_offset_1,
                self.frequency2 * 2 * np.pi * self.time_array + self.phase_offset_2,
                self.frequency3 * 2 * np.pi * self.time_array + self.phase_offset_3,
            ]
        )

    def do_clarke_transform(self):
        """Perform Clarke transform function.

        https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
        https://www.mathworks.com/help/physmod/sps/ref/clarketransform.html
        """
        # Clarke transform
        self.clarke_matrix = (2 / 3) * np.array(
            [
                [1, -(1 / 2), -(1 / 2)],
                [0, (np.sqrt(3) / 2), -(np.sqrt(3) / 2)],
                [(1 / 2), (1 / 2), (1 / 2)],
            ]
        )
        # Clarke transform function
        self.clarke_alpha_beta_zero_array = np.dot(
            self.clarke_matrix,
            np.cos(self.three_phase_data_matrix)
            * np.array([self.amplitude1, self.amplitude2, self.amplitude3])[:, None],
        )
        # assign outputs to individual variables for polar plots
        (
            self.clarke_alpha_instantaneous,
            self.clarke_beta_instantaneous,
            self.clarke_zero_instantaneous,
        ) = self.clarke_alpha_beta_zero_array[:, 0]
        # matplotlib likes positive vector lengths
        if self.clarke_alpha_instantaneous >= 0:
            self.clarke_alpha_instantaneous_angle = 0.0
        else:
            self.clarke_alpha_instantaneous_angle = np.pi
            self.clarke_alpha_instantaneous *= -1
        if self.clarke_beta_instantaneous >= 0:
            self.clarke_beta_instantaneous_angle = np.pi * 3 / 2
        else:
            self.clarke_beta_instantaneous_angle = np.pi / 2
            self.clarke_beta_instantaneous *= -1

    def do_park_transform(self):
        """Perform Park transform function.

        https://de.wikipedia.org/wiki/D/q-Transformation
        https://www.mathworks.com/help/physmod/sps/ref/clarketoparkangletransform.html
        """
        # create Park transformation matrix, with reference based on enum value
        if self.park_alignment == ParkAlignment.q_aligned:
            self.park_matrix = np.array(
                [
                    [
                        np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        -np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.zeros((self.sample_count)),
                    ],
                    [
                        np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.zeros((self.sample_count)),
                    ],
                    [
                        np.zeros((self.sample_count)),
                        np.zeros((self.sample_count)),
                        np.ones((self.sample_count)),
                    ],
                ]
            )
        elif self.park_alignment == ParkAlignment.d_aligned:
            self.park_matrix = np.array(
                [
                    [
                        np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.zeros((self.sample_count)),
                    ],
                    [
                        np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        -np.sin(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
                        np.zeros((self.sample_count)),
                    ],
                    [
                        np.zeros((self.sample_count)),
                        np.zeros((self.sample_count)),
                        np.ones((self.sample_count)),
                    ],
                ]
            )
        # perform the matrix math
        self.park_array = np.einsum(
            "ijk,ik->jk",
            self.park_matrix,
            self.clarke_alpha_beta_zero_array,
        )
        # save instantaneous values for polar plots
        (
            self.park_d_instantaneous,
            self.park_q_instantaneous,
            self.park_zero_instantaneous,
        ) = self.park_array[:, 0]
        # set the phase angle based on reference value
        if self.park_alignment == ParkAlignment.q_aligned:
            self.park_q_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0]
            self.park_d_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0] - np.pi / 2
        else:
            self.park_d_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0]
            self.park_q_instantaneous_angle = self.three_phase_data_matrix[PhaseIndex.Phase1, 0] + np.pi / 2
        # matplotlib likes positive vectors
        if self.park_d_instantaneous < 0:
            self.park_d_instantaneous *= -1
            self.park_d_instantaneous_angle += 2 * np.pi
        if self.park_q_instantaneous < 0:
            self.park_q_instantaneous *= -1
            self.park_q_instantaneous_angle += 2 * np.pi

    def pyplot_initial_plot(self):
        """Plot the data using matplotlib.pyplot."""
        # create figure
        self.figurex = plt.figure(constrained_layout=True)
        self.subplots = self.figurex.subfigures(2, 1)
        self.figure1 = self.subplots[0]
        self.figure2 = self.subplots[1]
        # add axes to figure
        self.axes = [
            self.figure1.add_subplot(1, 2, 1),
            self.figure1.add_subplot(1, 2, 2, polar=True),
            self.figure2.add_subplot(7, 1, 1),
            self.figure2.add_subplot(7, 1, 2),
            self.figure2.add_subplot(7, 1, 3),
            self.figure2.add_subplot(7, 1, 4),
            self.figure2.add_subplot(7, 1, 5),
            self.figure2.add_subplot(7, 1, 6),
            self.figure2.add_subplot(7, 1, 7),
        ]
        self.rotation_slider = Slider(
            ax=self.axes[2],
            label="Rotation (Rad)",
            valmin=0,
            valmax=1,
            valinit=self.phase_offset_1,
        )
        # register gui callbacks
        self.rotation_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.frequency1_slider = Slider(
            ax=self.axes[3],
            label="Phase1 Frequency (Hertz)",
            valmin=self.frequency1 - 1,
            valmax=self.frequency1 + 1,
            valinit=self.frequency1,
        )
        self.frequency1_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.frequency2_slider = Slider(
            ax=self.axes[4],
            label="Phase2 Frequency (Hertz)",
            valmin=self.frequency2 - 1,
            valmax=self.frequency2 + 1,
            valinit=self.frequency2,
        )
        self.frequency2_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.frequency3_slider = Slider(
            ax=self.axes[5],
            label="Phase3 Frequency (Hertz)",
            valmin=self.frequency3 - 1,
            valmax=self.frequency3 + 1,
            valinit=self.frequency3,
        )
        self.frequency3_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.amplitude1_slider = Slider(
            ax=self.axes[6],
            label="Phase1 Amplitude",
            valmin=0.5,
            valmax=1.5,
            valinit=1,
        )
        self.amplitude1_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.amplitude2_slider = Slider(
            ax=self.axes[7],
            label="Phase2 Amplitude",
            valmin=0.5,
            valmax=1.5,
            valinit=1,
        )
        self.amplitude2_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        self.amplitude3_slider = Slider(
            ax=self.axes[8],
            label="Phase3 Amplitude",
            valmin=0.5,
            valmax=1.5,
            valinit=1,
        )
        self.amplitude3_slider.on_changed(lambda x: self.pyplot_update_plots(x))
        # plot cartesian data
        self.reference_phase_line_plot = self.axes[0].plot(
            self.time_array,
            np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
            color="black",
            linewidth=15,
            alpha=0.15,
        )[0]
        self.phase1_line_plot = self.axes[0].plot(
            self.time_array,
            np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :]),
        )[0]
        self.phase2_line_plot = self.axes[0].plot(
            self.time_array,
            np.cos(self.three_phase_data_matrix[PhaseIndex.Phase2, :]),
        )[0]
        self.phase3_line_plot = self.axes[0].plot(
            self.time_array,
            np.cos(self.three_phase_data_matrix[PhaseIndex.Phase3, :]),
        )[0]
        self.clarke_alpha_line_plot = self.axes[0].plot(
            self.time_array,
            self.clarke_alpha_beta_zero_array[ClarkeIndex.Alpha, :],
            "--",
            linewidth=4,
        )[0]
        self.clarke_beta_line_plot = self.axes[0].plot(
            self.time_array,
            self.clarke_alpha_beta_zero_array[ClarkeIndex.Beta, :],
            "--",
            linewidth=4,
        )[0]
        self.park_d_line_plot = self.axes[0].plot(
            self.time_array,
            self.park_array[ParkIndex.D, :],
            ":",
            linewidth=4,
        )[0]
        self.park_q_line_plot = self.axes[0].plot(
            self.time_array,
            self.park_array[ParkIndex.Q, :],
            ":",
            linewidth=4,
        )[0]
        self.axes[0].legend(
            [
                "Ref",
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Clarke α",
                "Clarke β",
                "Park d",
                "Park q",
            ],
            loc="upper right",
        )
        # plot polar data
        radii = np.array([0, 1])
        self.reference_angle_polar_plot = self.axes[1].plot(
            [
                self.three_phase_data_matrix[PhaseIndex.Phase1, 0],
                self.three_phase_data_matrix[PhaseIndex.Phase1, 0],
            ],
            radii,
            color="black",
            linewidth=15,
            alpha=0.15,
        )[0]
        self.phase1_polar_plot = self.axes[1].plot(
            [
                self.three_phase_data_matrix[PhaseIndex.Phase1, 0],
                self.three_phase_data_matrix[PhaseIndex.Phase1, 0],
            ],
            radii,
        )[0]
        self.phase2_polar_plot = self.axes[1].plot(
            [
                self.three_phase_data_matrix[PhaseIndex.Phase2, 0],
                self.three_phase_data_matrix[PhaseIndex.Phase2, 0],
            ],
            radii,
        )[0]
        self.phase3_polar_plot = self.axes[1].plot(
            [
                self.three_phase_data_matrix[PhaseIndex.Phase3, 0],
                self.three_phase_data_matrix[PhaseIndex.Phase3, 0],
            ],
            radii,
        )[0]
        self.clarke_alpha_polar_plot = self.axes[1].plot(
            [0, 0],
            [0, self.clarke_alpha_instantaneous],
            "--",
            linewidth=4,
        )[0]
        self.clarke_beta_polar_plot = self.axes[1].plot(
            [(np.pi / 2), (np.pi / 2)],
            [0, self.clarke_beta_instantaneous],
            "--",
            linewidth=4,
        )[0]
        self.park_d_polar_plot = self.axes[1].plot(
            [self.park_d_instantaneous_angle, self.park_d_instantaneous_angle],
            [0, self.park_d_instantaneous],
            ":",
            linewidth=4,
        )[0]
        self.park_q_polar_plot = self.axes[1].plot(
            [self.park_q_instantaneous_angle, self.park_q_instantaneous_angle],
            [0, self.park_q_instantaneous],
            ":",
            linewidth=4,
        )[0]
        self.axes[1].legend(
            [
                "Ref",
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Clarke α",
                "Clarke β",
                "Park d",
                "Park q",
            ],
            loc="upper right",
        )
        self.figurex.canvas.set_window_title("Clarke & Park Transformation Demo")
        # interact!
        plt.show()

    def set_rotation_angle(self, phase_1_angle_radian: float):
        """Set the rotation angle.

        Args:
            phase_1_angle_radian: a number
        """
        # update values with fixed 120 degree offsets
        self.phase_offset_1 = phase_1_angle_radian * 2 * np.pi
        self.phase_offset_2 = self.phase_offset_1 + ((2 * np.pi) * (1 / 3))
        self.phase_offset_3 = self.phase_offset_2 + ((2 * np.pi) * (1 / 3))
        # "force" angles between 0 and 2*pi
        if self.phase_offset_1 < (2 * np.pi):
            self.phase_offset_1 += 2 * np.pi
        if self.phase_offset_2 > (2 * np.pi):
            self.phase_offset_2 -= 2 * np.pi
        elif self.phase_offset_2 < (2 * np.pi):
            self.phase_offset_2 += 2 * np.pi
        if self.phase_offset_3 > (2 * np.pi):
            self.phase_offset_3 -= 2 * np.pi
        elif self.phase_offset_3 < (2 * np.pi):
            self.phase_offset_3 += 2 * np.pi

    def set_frequency1(self, frequency: float):
        self.frequency1 = frequency

    def set_frequency2(self, frequency: float):
        self.frequency2 = frequency

    def set_frequency3(self, frequency: float):
        self.frequency3 = frequency

    def set_amplitude1(self, amplitude: float):
        self.amplitude1 = amplitude

    def set_amplitude2(self, amplitude: float):
        self.amplitude2 = amplitude

    def set_amplitude3(self, amplitude: float):
        self.amplitude3 = amplitude

    def pyplot_update_plots(self, val):
        """Callback function for updating matplotlib.pyplot graph data.

        Args:
            val ([type]): [description]
        """
        # update things
        self.set_rotation_angle(self.rotation_slider.val)
        self.set_frequency1(self.frequency1_slider.val)
        self.set_frequency2(self.frequency2_slider.val)
        self.set_frequency3(self.frequency3_slider.val)
        self.set_amplitude1(self.amplitude1_slider.val)
        self.set_amplitude2(self.amplitude2_slider.val)
        self.set_amplitude3(self.amplitude3_slider.val)
        # do math
        self.do_three_phase_time_domain_data()
        self.do_clarke_transform()
        self.do_park_transform()
        # update cartesian plots
        self.reference_phase_line_plot.set_ydata(
            self.amplitude1 * np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :])
        )
        self.phase1_line_plot.set_ydata(
            self.amplitude1 * np.cos(self.three_phase_data_matrix[PhaseIndex.Phase1, :])
        )
        self.phase2_line_plot.set_ydata(
            self.amplitude2 * np.cos(self.three_phase_data_matrix[PhaseIndex.Phase2, :])
        )
        self.phase3_line_plot.set_ydata(
            self.amplitude3 * np.cos(self.three_phase_data_matrix[PhaseIndex.Phase3, :])
        )
        self.clarke_alpha_line_plot.set_ydata(self.clarke_alpha_beta_zero_array[ClarkeIndex.Alpha, :])
        self.clarke_beta_line_plot.set_ydata(self.clarke_alpha_beta_zero_array[ClarkeIndex.Beta, :])
        self.park_d_line_plot.set_ydata(self.park_array[ParkIndex.D, :])
        self.park_q_line_plot.set_ydata(self.park_array[ParkIndex.Q, :])
        # update polar plots
        self.reference_angle_polar_plot.set_xdata(self.three_phase_data_matrix[PhaseIndex.Phase1, 0])
        self.reference_angle_polar_plot.set_ydata([0, self.clarke_alpha_instantaneous])
        self.phase1_polar_plot.set_xdata(self.three_phase_data_matrix[PhaseIndex.Phase1, 0])
        self.phase2_polar_plot.set_xdata(self.three_phase_data_matrix[PhaseIndex.Phase2, 0])
        self.phase3_polar_plot.set_xdata(self.three_phase_data_matrix[PhaseIndex.Phase3, 0])
        self.clarke_alpha_polar_plot.set_xdata(
            [self.clarke_alpha_instantaneous_angle, self.clarke_alpha_instantaneous_angle]
        )
        self.clarke_alpha_polar_plot.set_ydata([0, self.clarke_alpha_instantaneous])
        self.clarke_beta_polar_plot.set_xdata(
            [self.clarke_beta_instantaneous_angle, self.clarke_beta_instantaneous_angle]
        )
        self.clarke_beta_polar_plot.set_ydata([0, self.clarke_beta_instantaneous])
        self.park_d_polar_plot.set_xdata([self.park_d_instantaneous_angle, self.park_d_instantaneous_angle])
        self.park_d_polar_plot.set_ydata(
            [
                0,
                self.park_d_instantaneous,
            ]
        )
        self.park_q_polar_plot.set_xdata(
            [
                self.park_q_instantaneous_angle,
                self.park_q_instantaneous_angle,
            ]
        )
        self.park_q_polar_plot.set_ydata(
            [
                0,
                self.park_q_instantaneous,
            ]
        )
        # redraw figure
        self.figure1.canvas.draw_idle()


if __name__ == "__main__":
    ClarkeParkDemo()
