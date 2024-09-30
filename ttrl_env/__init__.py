import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.0"

try:
    from farama_notifications import notifications

    if "ttrl_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["ttrl_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_ttrl_envs():
    """Import the envs module so that envs register themselves."""

    from ttrl_env.envs.common.abstract import MultiAgentWrapper

    # intersection_env.py
    register(
        id="intersection-v0",
        entry_point="ttrl_env.envs.intersection_env:IntersectionEnv",
    )

    register(
        id="intersection-v1",
        entry_point="ttrl_env.envs.intersection_env:ContinuousIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v0",
        entry_point="ttrl_env.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="ttrl_env.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="ttrl_env.envs.roundabout_env:RoundaboutEnv",
    )

    # u_turn_env.py
    register(id="u-turn-v0", entry_point="ttrl_env.envs.u_turn_env:UTurnEnv")


_register_ttrl_envs()
