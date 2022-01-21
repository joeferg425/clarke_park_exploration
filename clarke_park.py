import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from enum import IntEnum


class ParkAlignment(IntEnum):
    q_aligned = 1
    d_aligned = 2


####################################################################
# configuration constants
frequency = 1
duration = 1
sample_rate = 100
sample_count = duration * sample_rate
park_alignment = ParkAlignment.d_aligned


####################################################################
# calculated constants
_2pi = 2 * np.pi
_120 = (1 / 3) * _2pi
phase_offset_1 = _120 * (0)
phase_offset_2 = _120 * (1)
phase_offset_3 = _120 * (2)


####################################################################
# three phase data
time_array = np.linspace(0, 1, sample_count)
phase1_data_array = frequency * 2 * np.pi * time_array + phase_offset_1
phase2_data_array = frequency * 2 * np.pi * time_array + phase_offset_2
phase3_data_array = frequency * 2 * np.pi * time_array + phase_offset_3
phase1_instantaneous = phase1_data_array[0]
phase2_instantaneous = phase2_data_array[0]
phase3_instantaneous = phase3_data_array[0]
three_phase_data_matrix = np.array(
    [
        np.cos(phase1_data_array),
        np.cos(phase2_data_array),
        np.cos(phase3_data_array),
    ]
)
instantaneous_three_phase_data_vector = np.array(
    [
        np.cos(phase1_data_array[0]),
        np.cos(phase2_data_array[0]),
        np.cos(phase3_data_array[0]),
    ]
)

####################################################################
# Clarke transform
clarke_matrix = (2 / 3) * np.array(
    [
        [1, -(1 / 2), -(1 / 2)],
        [0, (np.sqrt(3) / 2), -(np.sqrt(3) / 2)],
        [(1 / 2), (1 / 2), (1 / 2)],
    ]
)
# single-value'd Clarke transform on 0'th index of three-phase data
clarke_alpha_beta_zero_instantaneous = np.dot(
    clarke_matrix,
    instantaneous_three_phase_data_vector,
)
# assign outputs to individual variables as well
(
    clarke_alpha_instantaneous,
    clarke_beta_instantaneous,
    clarke_zero_instantaneous,
) = clarke_alpha_beta_zero_instantaneous
# Clarke transform along axis
clarke_alpha_beta_zero_array = np.dot(
    clarke_matrix,
    three_phase_data_matrix,
)
# assign outputs to individual variables as well
(
    clarke_alpha_array,
    clarke_beta_array,
    clarke_zero_array,
) = clarke_alpha_beta_zero_array

####################################################################
# Park transform
# create theta array
park_theta_array = phase1_data_array.copy()
park_theta_instantaneous = park_theta_array[0]
# create park transformation matrix
# assign outputs to individual variables as well
park_matrix = np.zeros([0])
if park_alignment == ParkAlignment.q_aligned:
    park_matrix = np.array(
        [
            [
                np.sin(park_theta_array),
                -np.cos(park_theta_array),
                np.zeros((sample_count)),
            ],
            [
                np.cos(park_theta_array),
                np.sin(park_theta_array),
                np.zeros((sample_count)),
            ],
            [
                np.zeros((sample_count)),
                np.zeros((sample_count)),
                np.ones((sample_count)),
            ],
        ]
    )
elif park_alignment == ParkAlignment.d_aligned:
    park_matrix = np.array(
        [
            [
                np.sin(park_theta_array),
                np.cos(park_theta_array),
                np.zeros((sample_count)),
            ],
            [
                np.cos(park_theta_array),
                -np.sin(park_theta_array),
                np.zeros((sample_count)),
            ],
            [
                np.zeros((sample_count)),
                np.zeros((sample_count)),
                np.ones((sample_count)),
            ],
        ]
    )

park_array = np.einsum(
    "ijk,ik->jk",
    park_matrix,
    clarke_alpha_beta_zero_array,
)
(
    park_d_instantaneous,
    park_q_instantaneous,
    park_zero_instantaneous,
) = park_array[:, 0]
park_d_instantaneous_angle = 0.0
park_q_instantaneous_angle = 0.0
if park_alignment == ParkAlignment.q_aligned:
    park_q_instantaneous_angle = phase1_instantaneous
    park_d_instantaneous_angle = phase1_instantaneous - np.pi / 2
else:
    park_d_instantaneous_angle = phase1_instantaneous
    park_q_instantaneous_angle = phase1_instantaneous + np.pi / 2
if park_d_instantaneous < 0:
    park_d_instantaneous *= -1
    park_d_instantaneous_angle += np.pi
if park_q_instantaneous < 0:
    park_q_instantaneous *= -1
    park_q_instantaneous_angle += np.pi

(
    park_d_array,
    park_q_array,
    park_zero_array,
) = park_array
####################################################################
# plot setup
# create figure
figure = plt.figure()
# add axes to figure
axes = [
    figure.add_subplot(1, 2, 1),
    figure.add_subplot(1, 2, 2, polar=True),
]
# make room for controls
plt.subplots_adjust(left=0.15, bottom=0.25)
axes_rotation = plt.axes([0.25, 0.1, 0.65, 0.03])
rotation_slider = Slider(
    ax=axes_rotation,
    label="Rotation (Rad)",
    valmin=0,
    valmax=1,
    valinit=phase_offset_1,
)
# plot cartesian data
reference_phase_line_plot = axes[0].plot(
    time_array,
    np.cos(phase1_data_array),
    color="black",
    linewidth=15,
    alpha=0.15,
)[0]
phase1_line_plot = axes[0].plot(
    time_array,
    np.cos(phase1_data_array),
)[0]
phase2_line_plot = axes[0].plot(
    time_array,
    np.cos(phase2_data_array),
)[0]
phase3_line_plot = axes[0].plot(
    time_array,
    np.cos(phase3_data_array),
)[0]
clarke_alpha_line_plot = axes[0].plot(
    time_array,
    clarke_alpha_array,
    "--",
    linewidth=4,
)[0]
clarke_beta_line_plot = axes[0].plot(
    time_array,
    clarke_beta_array,
    "--",
    linewidth=4,
)[0]
park_d_line_plot = axes[0].plot(
    time_array,
    park_d_array,
    ":",
    linewidth=4,
)[0]
park_q_line_plot = axes[0].plot(
    time_array,
    park_q_array,
    ":",
    linewidth=4,
)[0]
axes[0].legend(
    [
        "Ref",
        "Phase 1",
        "Phase2",
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
reference_angle_polar_plot = axes[1].plot(
    [phase1_instantaneous, phase1_instantaneous],
    radii,
    color="black",
    linewidth=15,
    alpha=0.15,
)[0]
phase1_polar_plot = axes[1].plot(
    [phase1_instantaneous, phase1_instantaneous],
    radii,
)[0]
phase2_polar_plot = axes[1].plot(
    [phase2_instantaneous, phase2_instantaneous],
    radii,
)[0]
phase3_polar_plot = axes[1].plot(
    [phase3_instantaneous, phase3_instantaneous],
    radii,
)[0]
clarke_alpha_polar_plot = axes[1].plot(
    [0, 0],
    [0, clarke_alpha_instantaneous],
    "--",
    linewidth=4,
)[0]
clarke_beta_polar_plot = axes[1].plot(
    [(np.pi / 2), (np.pi / 2)],
    [0, clarke_beta_instantaneous],
    "--",
    linewidth=4,
)[0]
park_d_polar_plot = axes[1].plot(
    [park_d_instantaneous_angle, park_d_instantaneous_angle],
    [0, park_d_instantaneous],
    ":",
    linewidth=4,
)[0]
park_q_polar_plot = axes[1].plot(
    [park_q_instantaneous_angle, park_q_instantaneous_angle],
    [0, park_q_instantaneous],
    ":",
    linewidth=4,
)[0]
axes[1].legend(
    [
        "Ref",
        "Phase 1",
        "Phase2",
        "Phase 3",
        "Clarke α",
        "Clarke β",
        "Park d",
        "Park q",
    ],
    loc="upper right",
)


# callback function for updating graph data
def update(val):
    phase_offset_1 = rotation_slider.val * 2 * np.pi
    phase_offset_2 = phase_offset_1 + ((2 * np.pi) * (1 / 3))
    phase_offset_3 = phase_offset_2 + ((2 * np.pi) * (1 / 3))
    # theta = phase_offset_1
    if phase_offset_3 > (2 * np.pi):
        phase_offset_3 -= 2 * np.pi
    phase1_data_array = frequency * 2 * np.pi * time_array + phase_offset_1
    phase2_data_array = frequency * 2 * np.pi * time_array + phase_offset_2
    phase3_data_array = frequency * 2 * np.pi * time_array + phase_offset_3
    three_phase_data_matrix = np.array(
        [
            np.cos(phase1_data_array),
            np.cos(phase2_data_array),
            np.cos(phase3_data_array),
        ]
    )
    phase1_instantaneous = phase1_data_array[0]
    phase2_instantaneous = phase2_data_array[0]
    phase3_instantaneous = phase3_data_array[0]
    instantaneous_three_phase_data_vector = np.array(
        [
            np.cos(phase1_instantaneous),
            np.cos(phase2_instantaneous),
            np.cos(phase3_instantaneous),
        ]
    )
    clarke_alpha_beta_zero_instantaneous = np.dot(
        clarke_matrix,
        instantaneous_three_phase_data_vector,
    )
    # assign outputs to individual variables as well
    (
        clarke_alpha_instantaneous,
        clarke_beta_instantaneous,
        clarke_zero_instantaneous,
    ) = clarke_alpha_beta_zero_instantaneous
    # Clarke transform along axis
    clarke_alpha_beta_zero_array = np.dot(
        clarke_matrix,
        three_phase_data_matrix,
    )
    # assign outputs to individual variables as well
    (
        clarke_alpha_array,
        clarke_beta_array,
        clarke_zero_array,
    ) = clarke_alpha_beta_zero_array
    park_theta_array = phase1_data_array.copy()
    if park_alignment == ParkAlignment.q_aligned:
        park_matrix = np.array(
            [
                [
                    np.sin(park_theta_array),
                    -np.cos(park_theta_array),
                    np.zeros((sample_count)),
                ],
                [
                    np.cos(park_theta_array),
                    np.sin(park_theta_array),
                    np.zeros((sample_count)),
                ],
                [
                    np.zeros((sample_count)),
                    np.zeros((sample_count)),
                    np.ones((sample_count)),
                ],
            ]
        )
    elif park_alignment == ParkAlignment.d_aligned:
        park_matrix = np.array(
            [
                [
                    np.sin(park_theta_array),
                    np.cos(park_theta_array),
                    np.zeros((sample_count)),
                ],
                [
                    np.cos(park_theta_array),
                    -np.sin(park_theta_array),
                    np.zeros((sample_count)),
                ],
                [
                    np.zeros((sample_count)),
                    np.zeros((sample_count)),
                    np.ones((sample_count)),
                ],
            ]
        )

    park_array = np.einsum(
        "ijk,ik->jk",
        park_matrix,
        clarke_alpha_beta_zero_array,
    )
    (
        park_d_instantaneous,
        park_q_instantaneous,
        park_zero_instantaneous,
    ) = park_array[:, 0]
    park_d_instantaneous_angle = 0
    park_q_instantaneous_angle = 0
    if park_alignment == ParkAlignment.q_aligned:
        park_q_instantaneous_angle = phase1_instantaneous
        park_d_instantaneous_angle = phase1_instantaneous - np.pi / 2
    else:
        park_d_instantaneous_angle = phase1_instantaneous
        park_q_instantaneous_angle = phase1_instantaneous + np.pi / 2
    if park_d_instantaneous < 0:
        park_d_instantaneous *= -1
        park_d_instantaneous_angle += np.pi
    if park_q_instantaneous < 0:
        park_q_instantaneous *= -1
        park_q_instantaneous_angle += np.pi

    reference_phase_line_plot.set_ydata(np.cos(phase1_data_array))
    phase1_line_plot.set_ydata(np.cos(phase1_data_array))
    phase2_line_plot.set_ydata(np.cos(phase2_data_array))
    phase3_line_plot.set_ydata(np.cos(phase3_data_array))
    clarke_alpha_line_plot.set_ydata(clarke_alpha_array)
    clarke_beta_line_plot.set_ydata(clarke_beta_array)
    park_d_line_plot.set_ydata(park_d_array)
    park_q_line_plot.set_ydata(park_q_array)

    reference_angle_polar_plot.set_xdata(phase1_instantaneous)
    phase1_polar_plot.set_xdata(phase1_instantaneous)
    phase2_polar_plot.set_xdata(phase2_instantaneous)
    phase3_polar_plot.set_xdata(phase3_instantaneous)
    # print(clarke_alpha_instantaneous)
    if clarke_alpha_instantaneous >= 0:
        clarke_alpha_polar_plot.set_xdata([0, 0])
    else:
        clarke_alpha_polar_plot.set_xdata([np.pi, np.pi])
    clarke_alpha_polar_plot.set_ydata([0, np.abs(clarke_alpha_instantaneous)])
    if clarke_beta_instantaneous >= 0:
        clarke_beta_polar_plot.set_xdata([(np.pi * 3 / 2), (np.pi * 3 / 2)])
    else:
        clarke_beta_polar_plot.set_xdata([(np.pi / 2), (np.pi / 2)])
    clarke_beta_polar_plot.set_ydata([0, np.abs(clarke_beta_instantaneous)])
    park_d_polar_plot.set_xdata(
        [park_d_instantaneous_angle, park_d_instantaneous_angle]
    )
    park_d_polar_plot.set_ydata(
        [
            0,
            park_d_instantaneous,
        ]
    )
    park_q_polar_plot.set_xdata(
        [
            park_q_instantaneous_angle,
            park_q_instantaneous_angle,
        ]
    )
    park_q_polar_plot.set_ydata(
        [
            0,
            park_q_instantaneous,
        ]
    )
    figure.canvas.draw_idle()


rotation_slider.on_changed(update)

plt.show()
