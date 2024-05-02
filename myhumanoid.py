import numpy as np
import os
import torch
import tensorflow as tf

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle, \
    get_euler_xyz

from isaacgymenvs.tasks.base.vec_task import VecTask

class MyHumanoid(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.all_rewards = []

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.posture_weight = self.cfg["env"]["postureWeight"]
        self.mismatch_weight = self.cfg["env"]["mismatchWeight"]
        self.velocity_variation_weight = self.cfg["env"]["velocityVariationWeight"]
        self.floor_contact_weight = self.cfg["env"]["floorContactWeight"]
        self.step_length_weight = self.cfg["env"]["stepLengthWeight"]
        self.front_foot_weight = self.cfg["env"]["frontFootWeight"]
        self.force_penalty_scale = self.cfg["env"]["forcePenaltyScale"]
        self.alive_reward = self.cfg["env"]["aliveReward"]


        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 120
        self.cfg["env"]["numActions"] = 21

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # self.velocities = torch.zeros_like(self.root_states[:, 7:10], device=self.device)
        self.prev_velocities = torch.zeros_like(self.root_states[:, 7:10], device=self.device)
        self.prev_left_foot_x = torch.zeros(self.num_envs, device=self.device)
        self.prev_right_foot_x = torch.zeros(self.num_envs, device=self.device)

        self.left_foot_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.right_foot_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # self.prev_torso = torch.zeros_like(self.root_states[:, 7:10], device=self.device)

        self.contact_threshold = 0.1
        self.swing_time_right = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.swing_time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.left_step_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.right_step_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.floor_contact = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.front_foot_timer = torch.full((self.num_envs,), 0.5, device=self.device)  # 0.5 second timer
        self.current_front_foot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # False: Left, True: Right
        self.reward_for_front_foot = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.force_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = ""

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        dof_dict = self.gym.get_asset_dof_dict(humanoid_asset)
        rb_dict = self.gym.get_asset_rigid_body_dict(humanoid_asset)

        self.left_hip_z_idx = dof_dict["left_hip_z"]
        self.right_hip_z_idx = dof_dict["right_hip_z"]

        self.left_hip_x_idx = dof_dict["left_hip_x"]
        self.right_hip_x_idx = dof_dict["right_hip_x"]

        self.abdomen_x_idx = dof_dict["abdomen_x"]
        self.abdomen_y_idx = dof_dict["abdomen_y"]
        self.abdomen_z_idx = dof_dict["abdomen_z"]

        self.left_ankle_x_idx = dof_dict["left_ankle_x"]

        self.right_ankle_x_idx = dof_dict["right_ankle_x"]

        # print("=====================================================================================")
        # print("# Rigid bodies: " + str(self.gym.get_asset_rigid_body_count(humanoid_asset)))
        # print("=====================================================================================")
        # rb_dict_list = sorted(rb_dict.items(), key=lambda item: item[1])
        # print(rb_dict_list)

        # print("=====================================================================================")
        # print("# DOFs: " + str(self.gym.get_asset_dof_count(humanoid_asset)))
        # print("=====================================================================================")
        # dof_dict_list = sorted(dof_dict.items(), key=lambda item: item[1])
        # print(dof_dict_list)
        # print("=====================================================================================")
        # print(TEST)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        self.right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        self.left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, self.right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, self.left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def save_rewards_to_csv(rewards, filename='rewards.csv'):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for reward in rewards:
                writer.writerow([reward])

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.all_rewards.append(self.rew_buf.mean().item())

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.left_foot_contact,
            self.right_foot_contact,
            self.swing_time_left,
            self.swing_time_right,
            self.posture_weight,
            self.mismatch_weight,
            self.velocity_variation_weight,
            self.floor_contact,
            self.floor_contact_weight,
            self.left_step_lengths,
            self.right_step_lengths,
            self.step_length_weight,
            self.reward_for_front_foot,
            self.front_foot_weight,
            self.force_penalty,
            self.alive_reward
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_force_tensor(self.sim)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_velocities[:] = compute_humanoid_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1, self.left_hip_z_idx, self.right_hip_z_idx, self.abdomen_x_idx, self.abdomen_y_idx, self.abdomen_z_idx, self.prev_velocities, self.rigid_body_state_tensor, self.left_ankle_x_idx, self.right_ankle_x_idx, self.left_hip_x_idx, self.right_hip_x_idx)

        # Extract foot forces
        right_foot_forces = self.vec_sensor_tensor[:, 0:3]
        left_foot_forces = self.vec_sensor_tensor[:, 6:9]

        # Determine contact (1 if in contact, 0 if not)
        new_right_contact = (right_foot_forces[:, 2] > self.contact_threshold).float()
        new_left_contact = (left_foot_forces[:, 2] > self.contact_threshold).float()

        new_right_contact = new_right_contact.bool()
        new_left_contact = new_left_contact.bool()

        current_left_foot_x = self.rigid_body_state_tensor[:, self.left_foot_idx, 0]
        current_right_foot_x = self.rigid_body_state_tensor[:, self.right_foot_idx, 0]

        self.left_step_lengths = torch.where(
            (self.left_foot_contact & ~new_left_contact),
            current_left_foot_x - self.prev_left_foot_x,
            torch.tensor(0.0, device=self.device)
        )

        self.right_step_lengths = torch.where(
            (self.right_foot_contact & ~new_right_contact),
            current_right_foot_x - self.prev_right_foot_x,
            torch.tensor(0.0, device=self.device)
        )

        # Update previous x-positions when the foot is in contact
        self.prev_left_foot_x = torch.where(new_left_contact, current_left_foot_x, self.prev_left_foot_x)
        self.prev_right_foot_x = torch.where(new_right_contact, current_right_foot_x, self.prev_right_foot_x)

        # Update contact states
        self.left_foot_contact = new_left_contact
        self.right_foot_contact = new_right_contact


        self.floor_contact = (self.right_foot_contact | self.left_foot_contact).float()

        # print(self.floor_contact)

        # Calculate swing time: Increment if not in contact, reset if in contact
        self.swing_time_right += (1 - self.right_foot_contact.float()) * self.dt
        self.swing_time_left += (1 - self.left_foot_contact.float()) * self.dt

        # Determine which foot is currently in front
        new_front_foot = current_right_foot_x > current_left_foot_x  # True if right foot is in front

        # Update timer based on change of front foot
        foot_changed = new_front_foot != self.current_front_foot
        self.front_foot_timer[foot_changed] = 0.5  # Reset timer to 0.5 seconds when foot changes
        self.front_foot_timer[~foot_changed] -= self.dt  # Decrease timer

        # Clamp timer to not go below zero
        self.front_foot_timer.clamp_(min=0)

        # Update the current front foot tracker
        self.current_front_foot = new_front_foot

        self.reward_for_front_foot = torch.where(
            self.front_foot_timer > 0,
            torch.tensor(1, device=self.device),  # Reward if timer > 0
            torch.tensor(-1, device=self.device)  # Penalty if timer == 0
        )

        dof_forces = self.dof_force_tensor
        self.force_penalty = torch.sum(dof_forces ** 2, dim=1) * self.force_penalty_scale


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_humanoid_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    max_motor_effort,
    motor_efforts,
    termination_height,
    death_cost,
    max_episode_length,
    left_foot_contact,
    right_foot_contact,
    swing_time_left,
    swing_time_right,
    posture_weight,
    mismatch_weight,
    velocity_variation_weight,
    floor_contact,
    floor_contact_weight,
    left_step_lengths,
    right_step_lengths,
    step_length_weight,
    front_foot_reward,
    front_foot_weight,
    force_penalty,
    alive_reward
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float, Tensor, Tensor, Tensor, Tensor, float, float, float, Tensor, float, Tensor, Tensor, float, Tensor, float, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # reward for being upright
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    actions_cost = torch.sum(actions ** 2, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    alive_reward = torch.ones_like(potentials) * alive_reward
    progress_reward = potentials - prev_potentials

    # step mismatch penalty
    mismatch_penalty = mismatch_weight * ((swing_time_left - swing_time_right) ** 2)

    # Posture deviation penalty components
    posture_penalty = torch.zeros_like(heading_reward)
    left_hip_yaw_penalty = torch.where(obs_buf[:, 81] > 0, obs_buf[:, 81] ** 2, posture_penalty)
    right_hip_yaw_penalty = torch.where(obs_buf[:, 82] > 0, obs_buf[:, 82] ** 2, posture_penalty)
    abdomen_roll_penalty = torch.where(obs_buf[:, 83] > 0, obs_buf[:, 83] ** 2, posture_penalty)
    abdomen_pitch_penalty = torch.where(obs_buf[:, 84] > 0, obs_buf[:, 84] ** 2, posture_penalty)
    abdomen_yaw_penalty = torch.where(obs_buf[:, 85] > 0, obs_buf[:, 85] ** 2, posture_penalty)
    left_ankle_roll_penalty = torch.where(obs_buf[:, 89] > 0, obs_buf[:, 89] ** 2, posture_penalty)
    right_ankle_roll_penalty = torch.where(obs_buf[:, 90] > 0, obs_buf[:, 90] ** 2, posture_penalty)
    left_hip_roll_penalty = torch.where(obs_buf[:, 91] > 0, obs_buf[:, 91] ** 2, posture_penalty)
    right_hip_roll_penalty = torch.where(obs_buf[:, 92] > 0, obs_buf[:, 92] ** 2, posture_penalty)
    posture_penalty = posture_weight * (left_hip_yaw_penalty + right_hip_yaw_penalty + abdomen_roll_penalty + abdomen_pitch_penalty + abdomen_yaw_penalty + left_ankle_roll_penalty + \
    right_ankle_roll_penalty + left_hip_roll_penalty + right_hip_roll_penalty)

    # velocity variation penalty
    velocity_variation_penalty = velocity_variation_weight * obs_buf[:, 86]

    # floor contact reward
    floor_contact_reward = floor_contact * floor_contact_weight

    # step length reward
    step_length_reward = step_length_weight * (left_step_lengths * left_foot_contact.float() + right_step_lengths + right_foot_contact.float())

    # foot alternation reward

    foot_alt_reward = front_foot_reward * front_foot_weight

    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - \
        dof_at_limit_cost - posture_penalty - mismatch_penalty - velocity_variation_penalty + \
        floor_contact_reward + step_length_reward + foot_alt_reward - force_penalty

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # print(total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1, left_hip_z_idx, right_hip_z_idx, abdomen_x_idx, abdomen_y_idx, abdomen_z_idx, prev_velocities, rigid_body_states, left_ankle_x_idx, right_ankle_x_idx, left_hip_x_idx, right_hip_x_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor, int, int, int, int, int, Tensor, Tensor, int, int, int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:,  0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    left_hip_yaw = dof_pos[:, left_hip_z_idx]
    right_hip_yaw = dof_pos[:, right_hip_z_idx]

    left_hip_roll = dof_pos[:, left_hip_x_idx]
    right_hip_roll = dof_pos[:, right_hip_x_idx]

    left_hip_yaw = normalize_angle(left_hip_yaw).unsqueeze(-1)
    right_hip_yaw = normalize_angle(right_hip_yaw).unsqueeze(-1)

    left_hip_roll = normalize_angle(left_hip_roll).unsqueeze(-1)
    right_hip_roll = normalize_angle(right_hip_roll).unsqueeze(-1)

    abdomen_roll = dof_pos[:, abdomen_x_idx]
    abdomen_pitch = dof_pos[:, abdomen_y_idx]
    abdomen_yaw = dof_pos[:, abdomen_z_idx]

    left_ankle_roll = dof_pos[:, left_ankle_x_idx]

    right_ankle_roll = dof_pos[:, right_ankle_x_idx]

    left_ankle_roll = normalize_angle(left_ankle_roll).unsqueeze(-1)

    right_ankle_roll = normalize_angle(right_ankle_roll).unsqueeze(-1)

    abdomen_roll = normalize_angle(abdomen_roll).unsqueeze(-1)
    abdomen_pitch = normalize_angle(abdomen_pitch).unsqueeze(-1)
    abdomen_yaw = normalize_angle(abdomen_yaw).unsqueeze(-1)

    delta_velocity = root_states[:, 7:10] - prev_velocities
    prev_velocities_new = velocity.clone()

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21), 1, 1, 1, 1, 1, 3, 1, 1, 1, 1
    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions, left_hip_yaw, right_hip_yaw, abdomen_roll, abdomen_pitch, abdomen_yaw, delta_velocity, left_ankle_roll, right_ankle_roll, left_hip_roll, right_hip_roll), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec, prev_velocities_new
