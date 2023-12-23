import numpy as np

class create_function:
    def periodic_kernel(self, t1, t2, tau_target):
        return np.exp(-2 * np.sin(np.pi * np.abs(t1 - t2) / tau_target)**2)

    def sample_from_gp(self, t_grid, tau_target, num_samples):
        cov_matrix = self.periodic_kernel(t_grid[:, None], t_grid[:, None].T, tau_target)
        samples = np.random.multivariate_normal(mean=np.zeros_like(t_grid), cov=cov_matrix, size=num_samples)
        return t_grid, samples