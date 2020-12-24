from dataclasses import dataclass
from typing import Any

@dataclass
class State:
    img: Any = None # used by learned model, not needed for ground truth model
    robot: Any = None # robot eef pos, used by learned model, not needed for gt model
    mask: Any = None #  used by learned model, not needed for ground truth model
    sim: Any = None # used by learned model, not needed for ground truth model

@dataclass
class DemoGoalState:
    imgs: Any = None # list of goal imgs for computing costs
    robots: Any = None # list of goal eef pos
    masks: Any = None # list of goal masks