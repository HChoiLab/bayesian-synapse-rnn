import numpy as np


class RNN:
    def __init__(self, p, g, N, tau_m, tau_target):
        self.J = self.sample_recurrent_weights(p, g, N)
        self.A = self.sample_feedback_weights(N)
        self.w = self.initialize_readout_weights(N)
        # self.input_time, self.amplitude = self.sample_inputs(N)
        self.V = np.zeros(N)
        self.t = 0
        self.dt = 1
        self.tau_m = tau_m
        self.tau_target = tau_target


    def sample_recurrent_weights(self, p, g, N):
        bernoulli_random = np.random.binomial(1, p, (N, N))
        J0 = np.random.normal(0, (g**2)/(p*N), (N, N))
        J = np.multiply(bernoulli_random, J0)
        return J

    def sample_feedback_weights(self, N):
        A = np.random.uniform(-1, 1, N)
        return A
    
    def initialize_readout_weights(self, N):
        w = np.random.normal(0, 1/N, N)
        return w

    def sample_inputs(self, duration_ms=1, sampling_rate=10):
        np.random.seed(42)
        input_time = np.linspace(0, duration_ms, int(duration_ms * sampling_rate), endpoint=False)
        input_amplitude = np.random.uniform(-1, 1, len(input_time))
        return input_time, input_amplitude
    
    def error(self, V_target, V):
        return (V_target - V)

    def forward_pass(self, t, input_time, input_amplitude): 
        
        if t >= len(input_time):
            input = 0
        elif t < 0:
            input = 0
        else:
            input = input_amplitude[t]
        self.V = self.V + self.dt * (-self.V + np.dot(self.J, np.tanh(self.V)) + self.A * self.V + input) / self.tau_m
        out = np.dot(self.w, self.V)
        return out