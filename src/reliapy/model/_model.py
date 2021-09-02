import numpy as np


class StateLimit:

    def __init__(self, state_limit_function=None, system=False, num_sle=1):

        if callable(state_limit_function):
            self.state_limit_function = state_limit_function
        else:
            raise TypeError('reliapy: state_limit_equations must be callable.')

        self.system = system
        self.num_sle = num_sle
        self.g = []

    def run(self, samples=None):

        # Check if samples are provided
        if samples is None:
            raise ValueError('reliapy: Samples must be provided as input.')
        elif isinstance(samples, list):
            nsim = len(samples)  # This assumes that the number of rows is the number of simulations.
        else:
            raise ValueError('reliapy: Samples must be passed as a list')

        # Run python model
        g = []
        for i in range(nsim):

            state_lim = self.state_limit_function(samples[i])

            if self.system:
                # state_lim is a list [g_total, g0, g1, g2,...]
                if len(state_lim) != self.num_sle:
                    raise ValueError('reliapy: num_sle is not consistent with the number of State Limit Equations.')

                if not isinstance(state_lim, list):
                    raise TypeError('reliapy: The output of the state limit must be a list')

            g.append(state_lim)

        return g
