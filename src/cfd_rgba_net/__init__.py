"""cfd-rgba-net: learned per-voxel flow-feature probabilities for CFD volume rendering."""

__version__ = "0.1.0"

# Version of the input contract (the meaning, order and units of the model's input channels).
# Any change to preprocessing that changes the numbers a model sees must bump it. Contract.load
# refuses a contract YAML whose contract_version differs, so stale configs fail loudly.
CONTRACT_VERSION = "1.0"

# Version of the label rules (configs/label_rules_v1.yaml + cfd_rgba_net.label). LabelRules.load
# refuses a YAML whose label_rules_version differs, for the same reason.
LABEL_RULES_VERSION = "1"
