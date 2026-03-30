class HexapodEnv(DirectRLEnv):
    """Rest of the environment definition omitted."""
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, HexapodRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
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
            ],
            dim=-1,
        )
        observations = {"policy":obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
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