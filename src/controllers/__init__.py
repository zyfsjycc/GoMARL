REGISTRY = {}

from .basic_controller import BasicMAC
from .group_controller import NMAC as GroupMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["group_mac"] = GroupMAC
