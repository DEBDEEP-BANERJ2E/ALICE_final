# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ALICE RL Environment.

ALICE (Adversarial Loop for Inter-model Co-evolutionary Environment) is an
OpenEnv-compliant RL training environment for code generation agents.
"""

from typing import Any, Dict, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class AliceAction(Action):
    """
    Action for the ALICE environment - agent's code solution attempt.
    
    The action is a string containing the agent's code solution to the
    current task. It will be verified through a 3-tier verification stack.
    """

    code: str = Field(..., description="Agent's code solution to the task")
    
    @field_validator("code")
    @classmethod
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("code must be non-empty")
        return v


class AliceObservation(Observation):
    """
    Observation from the ALICE environment after processing an action.
    
    Contains the current state, reward, done flag, and detailed info about
    verification results, turn number, and task context.
    """

    task: str = Field(default="", description="Current coding task prompt")
    turn_number: int = Field(default=1, description="Current turn (1-3)")
    feedback: str = Field(default="", description="Feedback from previous turn")
    hint: Optional[str] = Field(default=None, description="Hint provided on turn 3")
    verification_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verification results from 3-tier stack"
    )
    task_difficulty: float = Field(default=0.0, description="Task difficulty score (0-100)")
    discrimination_coverage: float = Field(
        default=0.0,
        description="Curriculum discrimination zone coverage"
    )
