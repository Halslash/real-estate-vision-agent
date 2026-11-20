"""Member A: CNN-based visual-quality proxy scoring for real-estate images.

The public integration surface for Member B is :class:`src.vision_agent.VisionAgent`.
The target is a market-derived visual proxy, not a physical-condition label and
not a price estimate.
"""

__all__ = [
    "dataset",
    "model",
    "plots",
    "train",
    "evaluate",
    "probe",
    "inspect_csv",
    "vision_agent",
    "losses",
    "data_audit",
    "make_folds",
    "oof_contract",
    "oof_predict",
    "demonstrate",
    "run_experiment",
]
