"""
Purpose: Generate deterministic open-loop action sequences that trigger ontology perception contacts and motion patterns.
Main contents: G1 joint-index caching and a staged scripted policy for crouch, arm-sweep, and step-probe motions.
"""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation


class ScriptedExplorer:
    """Open-loop action generator for the G1 ontology perception task."""

    def __init__(self, env):
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        self.device = env.device
        self.action_dim = self.robot.num_joints
        self.left_hip_pitch_ids = self.robot.find_joints("left_hip_pitch_joint")[0]
        self.right_hip_pitch_ids = self.robot.find_joints("right_hip_pitch_joint")[0]
        self.knee_ids = self.robot.find_joints(".*_knee_joint")[0]
        self.ankle_pitch_ids = self.robot.find_joints(".*_ankle_pitch_joint")[0]
        self.left_shoulder_pitch_ids = self.robot.find_joints("left_shoulder_pitch_joint")[0]
        self.right_shoulder_pitch_ids = self.robot.find_joints("right_shoulder_pitch_joint")[0]
        self.left_shoulder_roll_ids = self.robot.find_joints("left_shoulder_roll_joint")[0]
        self.right_shoulder_roll_ids = self.robot.find_joints("right_shoulder_roll_joint")[0]
        self.elbow_ids = self.robot.find_joints(".*_elbow_joint")[0]
        self.wrist_roll_ids = self.robot.find_joints(".*_wrist_roll_joint")[0]
        self.waist_pitch_ids = self.robot.find_joints("waist_pitch_joint")[0]

    def _zeros(self, num_envs: int) -> torch.Tensor:
        return torch.zeros((num_envs, self.action_dim), device=self.device, dtype=torch.float32)

    def action(self, step_index: int, num_envs: int) -> torch.Tensor:
        actions = self._zeros(num_envs)

        if step_index < 40:
            return actions

        if step_index < 120:
            phase = (step_index - 40) / 80.0
            crouch = math.sin(min(phase, 1.0) * math.pi)
            actions[:, self.left_hip_pitch_ids] = -0.20 * crouch
            actions[:, self.right_hip_pitch_ids] = -0.20 * crouch
            actions[:, self.knee_ids] = 0.45 * crouch
            actions[:, self.ankle_pitch_ids] = -0.20 * crouch
            actions[:, self.waist_pitch_ids] = 0.08 * crouch
            actions[:, self.elbow_ids] = 0.10 * crouch
            return actions

        if step_index < 220:
            phase = (step_index - 120) / 100.0
            sweep = math.sin(phase * math.pi)
            actions[:, self.left_shoulder_pitch_ids] = 0.25 * sweep
            actions[:, self.right_shoulder_pitch_ids] = 0.10 * sweep
            actions[:, self.left_shoulder_roll_ids] = 0.40 * sweep
            actions[:, self.right_shoulder_roll_ids] = -0.10 * sweep
            actions[:, self.elbow_ids] = 0.20 * sweep
            actions[:, self.wrist_roll_ids] = 0.15 * sweep
            return actions

        if step_index < 320:
            phase = (step_index - 220) / 100.0
            sweep = math.sin(phase * math.pi)
            actions[:, self.left_shoulder_pitch_ids] = 0.10 * sweep
            actions[:, self.right_shoulder_pitch_ids] = 0.25 * sweep
            actions[:, self.left_shoulder_roll_ids] = 0.10 * sweep
            actions[:, self.right_shoulder_roll_ids] = -0.40 * sweep
            actions[:, self.elbow_ids] = 0.20 * sweep
            actions[:, self.wrist_roll_ids] = -0.15 * sweep
            return actions

        phase = (step_index - 320) / 100.0
        gait = math.sin(phase * 2.0 * math.pi)
        actions[:, self.left_hip_pitch_ids] = -0.10 * gait
        actions[:, self.right_hip_pitch_ids] = 0.10 * gait
        actions[:, self.knee_ids[:1]] = 0.18 * max(gait, 0.0)
        actions[:, self.knee_ids[1:]] = 0.18 * max(-gait, 0.0)
        actions[:, self.ankle_pitch_ids[:1]] = -0.08 * max(gait, 0.0)
        actions[:, self.ankle_pitch_ids[1:]] = -0.08 * max(-gait, 0.0)
        actions[:, self.left_shoulder_pitch_ids] = 0.10 * abs(gait)
        actions[:, self.right_shoulder_pitch_ids] = 0.10 * abs(gait)
        return actions
