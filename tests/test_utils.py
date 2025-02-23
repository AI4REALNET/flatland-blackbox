class MockAgent:
    """
    Simplified stand-in for Agent class where
    only the attributes used by the solver are defined.
    """

    def __init__(self, handle, initial_position, target):
        self.handle = handle
        self.initial_position = initial_position  # (row, col)
        self.target = target  # (row, col)
