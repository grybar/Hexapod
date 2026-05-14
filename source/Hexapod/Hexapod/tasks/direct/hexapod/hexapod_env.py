# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from isaaclab_assets.robots.hexapod import HEXAPOD_CFG


@configclass
class HexapodEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )
    # robot
    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [15] * 12

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.31

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1

class HexapodEnv(LocomotionEnv):
    cfg: HexapodEnvCfg

    def __init__(self, cfg: HexapodEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # World-space target direction: travel along +Y.
        self.targets = torch.tensor(
            [0.0, 1000.0, 0.0],
            dtype=torch.float32,
            device=self.sim.device,
        ).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins

        # Robot-local forward direction: treat local +Y as "forward".
        self.heading_vec = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=torch.float32,
            device=self.sim.device,
        ).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
