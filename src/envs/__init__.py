from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
    from .grf import (Academy_3_vs_1_with_Keeper, 
                      Academy_Counterattack_Easy,
                      Academy_Pass_and_Shoot_with_Keeper)
except:
    gfootball = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
    REGISTRY["academy_3_vs_1_with_keeper"] = partial(env_fn, env=Academy_3_vs_1_with_Keeper)
    REGISTRY["academy_counterattack_easy"] = partial(env_fn, env=Academy_Counterattack_Easy)
    REGISTRY["academy_pass_and_shoot_with_keeper"] = partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
