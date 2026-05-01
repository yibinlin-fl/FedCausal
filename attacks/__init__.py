"""Attack simulation components."""

from attacks.label_flip import flip_labels, save_malicious_client_ids, select_malicious_clients
from attacks.prototype_scaling import apply_prototype_scaling_to_payload, scale_prototypes
from attacks.visualization import save_attack_visualizations

__all__ = [
    "apply_prototype_scaling_to_payload",
    "flip_labels",
    "save_malicious_client_ids",
    "save_attack_visualizations",
    "scale_prototypes",
    "select_malicious_clients",
]
