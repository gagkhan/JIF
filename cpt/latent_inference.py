class LatentInference:

    def __init__(self) -> None:
        self.encoder = None
        self.invdyn = None  # inverse dynamics, or just the regular policy
        self.fwddyn = None

    def forward(self, o_curr, o_next, o_goal):
        """
        === ILPO ===

        x_curr = self.encoder(o_curr)
        x_next = self.encoder(o_next)
        x_goal = self.encoder(o_goal)

        z_curr = self.invdyn(x_goal, x_curr)
        x_next_pred = self.fwddyn(x_curr, z_curr)

        === LAPO ===

        x_curr = self.encoder(o_curr)
        x_next = self.encoder(o_next)
        x_goal = self.encoder(o_goal)

        z_curr = self.invdyn(x_next, x_curr)
        x_next_pred = self.fwddyn(x_curr, z_curr)

        Notes: The only difference seems to be in this line where invdyn is called
        z_curr = self.invdyn(x_next or x_goal, x_curr). It would be perhaps be remove o_next as
        input and pass o_goal or o_goal based on whether this is ILPO or LAPO

        """

        x_curr = None
        x_next = None
        x_next_pred = None
        z_curr = None
        x_goal = None

        result = {
            "x_next_pred": x_next_pred,
            "z_curr": z_curr,
            "x_next": x_next,
            "x_curr": x_curr,
            "x_goal": x_goal,
        }

        return result
