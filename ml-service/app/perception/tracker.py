"""Phase 4 — video object tracking (persists object identity across frames
so temporal reasoning, e.g. 'this hazard has persisted for 3 minutes', is
possible). Not used in the Phase 1 single-image path.
"""


class Tracker:
    def __init__(self):
        raise NotImplementedError("Object tracking is a Phase 4 feature (PRD §16).")