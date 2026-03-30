# Copyright (c) 2025, Hexapod Lab Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

import isaaclab.sim as sim_utils

from isaaclab.sim import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg, UsdFileCfg, DomeLightCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.sensors import RayCasterCfg

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

HEXAPOD_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*hip", ".*knee", ".*ankle"],
    saturation_effort=1.0, 
    effort_limit=0.5,  
    velocity_limit=3.0,  
    stiffness={".*": 10.0},
    damping={".*": 0.3},
)

HEXAPOD_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="""D:\models\hexapod.usd""",
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            ".*hip": 0.0,
            ".*knee": -0.7,
            ".*ankle": 2.27,
        },
    ),
    actuators={"legs": HEXAPOD_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)