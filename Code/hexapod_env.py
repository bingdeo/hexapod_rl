from __future__ import annotations
#__future__ : for importing features from future version of python 
# annotations : making interpreters delay evaluating class
# -> we can use undefined class as tpye hint

import gymnasium as gym
#Provide API. creating, interacting with reinforcements environments.
import torch
#torch: machine learning framework. Support tensor computation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
#isaaclab.assets: part loading models and sensors
#Articulation: class load, control, simulate multi-joints robot
from isaaclab.envs import DirectRLEnv
from .hexapod_env_cfg import HexapodRoughEnvCfg
from isaaclab.sensors import ContactSensor, RayCaster

class HexapodEnv(DirectRLEnv):
    cfg: HexapodRoughEnvCfg

    def __init__(self, cfg: HexapodRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            #keys_part
        }
        # get specific body indices
        self._base_id, _=self._contact_sensor.find_bodies("body")
        #I have to check it
        self._toe_ids, _=self._contact_sensor.find_bodies(".*toe")
        #we have to add more undesired parts, but we have to test first
  
        self._undesired_contact_body_ids, _=self._contact_sensor.find_bodies([".*foot","body"])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        #Make sensor
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        #register sensor in simulation, scene.sensors areContact Sensor

        if isinstance(self.cfg, HexapodRoughEnvCfg):
         #  we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate?
        self.scene.clone_environments(copy_from_source=False)
        
        #add lights
        light_cfg = sim_utils.DomeLightCfg(intensity = 2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, HexapodRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
        #torch.cat: combine tensors along specific dimension
        #using for combine diverse state as one observation vector
            [
                tensor
                    for tensor in (
	                    self._robot.data.root_lin_vel_b,
	                    self._robot.data.root_ang_vel_b,
	                    self._robot.data.projected_gravity_b,
	                    self._commands,
	                    self._robot.data.joint_pos-self._robot.data.default_joint_pos,
	                    self._robot.data.joint_vel,
	                    height_data,
	                    self._actions,
	                )
	                if tensor is not None
	                #this is the problem of runtime error
            ],
            dim=-1,
        )
        observations = {"policy":obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(0.0 - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._toe_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._toe_ids]
        air_time = torch.sum((last_air_time-0.5)*first_contact, dim=1)*(
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        #undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=1), dim=1)[0] > 1.0
        )
        undesired_contacts = torch.sum(is_contact, dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        #rewards_part
        
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
        

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] >1000.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
#Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()

        total_episode_reward = 0.0

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            
            total_episode_reward += episodic_sum_avg

            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras=dict()

        extras["Episode_Reward/total"] = total_episode_reward / self.max_episode_length_s
        extras["Episode_Termination/base_contact"]=torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"]=torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)