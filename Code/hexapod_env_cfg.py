# Copyright (c) 2025, Hexapod Lab Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
#randomize state transition for preventing overfitting
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
#environments setting
#DirectRLEnv: recieve action, observe and compute 
from isaaclab.managers import EventTermCfg as EventTerm
#managers: make environments behavior module
#EventTerm: randomize mass, friction. Diffrent with mdp
from isaaclab.managers import SceneEntityCfg
#setting for designating name of parts of robot
from isaaclab.scene import InteractiveSceneCfg
#forming entire simulation
#for contact_sensor
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

#code that we were defined
from isaaclab_assets.robots.hexapod import HEXAPOD_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

class EventCfg:

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        #func: called when event happen
        mode="startup",
        #startup: our settings operating when simulation start
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            #designate the body that we make event
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            #divide range of the values that we randomize
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"), 
            #only affect mass
            "mass_distribution_params": (-0.05, 0.05),
            #Modify 0.5->0.05. 0.5kg is too big for our model
            "operation": "add",
            #add mass to the existing mass
        },
    )

@configclass
class HexapodRoughEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 18  # 6legs * 3joints
    observation_space = 253
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        #step
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(8.0, 8.0),
        ),
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        #We can simulate max 1024 in parallel
        env_spacing=3.0,
        #distance between each environments
        replicate_physics=True
        #Each env is treated as same
    )

    events: EventCfg = EventCfg()

    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #about sensor code
    contact_sensor : ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.0155)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    