import numpy as np
import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    def __init__(self, args, logger=print):
        super().__init__()
        self.channel_info = args.channel_ndg_info
        self.channel_noise_mean = 0
        self.channel_noise_std = self.channel_info['sigma_noise']
        self.p_constraint = torch.tensor(args.channel_ndg_info['ndg']['constraint_value'])  # power constraint of NDG
        if args.x_dim == args.y_dim:
            self.dim = args.y_dim
            self.capacity_gt = self.dim * 0.5 * np.log(1 + self.p_constraint / self.channel_noise_std)

        logger.info('| AWGN channel with dimension {} and SNR {} dB (at maximum capacity!)'.format(self.dim,
                    round(10 * np.log10(self.p_constraint.item() / self.channel_noise_std), 2)))

    def forward(self, x):
        """
        Simulate the transmission of the signal x through the channel.
        This method is automatically called when you apply the module to an input tensor.
        """
        noise = torch.zeros(x.size())
        for i in range(x.size(1)):
            noise[:, i] = torch.normal(self.channel_noise_mean, np.sqrt(self.channel_noise_std / x.size(1)), size=[x.size(0)], dtype=x.dtype)
        noise = noise.to(x.device)
        return x + (noise).detach()

    def erase_states(self):
        pass


class GMA1Channel(nn.Module):
    def __init__(self, args, logger=print):
        super().__init__()
        self.channel_info = args.channel_ndg_info
        self.channel_feedback = self.channel_info['channel_feedback']
        self.channel_noise_mean = 0
        self.channel_noise_std = self.channel_info['sigma_noise']
        self.alpha = self.channel_info['alpha']
        self.p_constraint = torch.tensor(args.channel_ndg_info['ndg']['constraint_value'])  # power constraint of NDG
        if args.x_dim == args.y_dim:
            self.dim = args.y_dim
            self.capacity_gt = self.calculate_capacity(self.p_constraint, self.channel_noise_std, self.alpha, feedback=self.channel_feedback)
            if self.capacity_gt is None:
                del self.capacity_gt
        self.previous_noise = None
        logger.info('| GMA(1) channel with dimension {} and SNR {} dB (at maximum capacity!)'.format(self.dim,
                    round(10 * np.log10(self.p_constraint.item() / self.channel_noise_std), 2)))

    @staticmethod
    def water_filling(noise_frequency_power_vec, p_condition, tol):
        K = len(noise_frequency_power_vec)  # K is the number of frequency slices

        # Initial step
        water_line = np.min(noise_frequency_power_vec) + p_condition / K  # Initialize waterline
        omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
        initial_P_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
        ptot = (1 / (2 * np.pi)) * np.trapz(initial_P_vec, omega_vec)  # Calculate total power for current waterline

        # Iteratively perform water filling
        while abs(p_condition - ptot) > tol:  # Continue as long as the error is higher than the given tolerance
            water_line = water_line + (p_condition - ptot) / K  # Raise water line
            p_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
            ptot = (1 / (2 * np.pi)) * np.trapz(p_vec, omega_vec)  # Compute new total power

        return water_line

    def calculate_capacity(self, P, N, alpha, feedback=True):
        if feedback:
            # https://arxiv.org/pdf/cs/0411036.pdf
            P = P / N  # to adjust P when N is not 1
            coeff = [-np.abs(alpha) ** 2,
                     2 * np.abs(alpha),
                     np.abs(alpha) ** 2 - 1 - P,
                     -2 * np.abs(alpha),
                     1]
            roots = np.roots(coeff)
            x0 = roots[np.all([roots.real >= 0, np.isreal(roots)], axis=0)].real[0]
            capacity = -np.log(x0)
        else:
            P = P/N         # to adjust P when N is not 1
            K = 10000
            tolerance = 1e-6
            # Get K slices of the PSD of Z_i
            omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
            H_w = 1 + alpha ** 2 + 2 * alpha * np.cos(omega_vec)
            # Calculate water line
            water_line = self.water_filling(H_w, P.item(), tolerance)
            # Calculate power allocation for each water line outcome
            P_allocation = np.maximum(0, water_line - H_w)
            # Compute allocation error (sanity check)
            allocation_error = np.sum(P_allocation - P.item())
            # Calculate capacity for the single value of P
            ones_vec = np.ones(K)
            int_vec = 0.5 * np.log(ones_vec + (1/self.dim) * P_allocation / H_w)
            capacity = self.dim * (1 / (2 * np.pi)) * np.trapz(int_vec, omega_vec)
        return capacity

    def forward(self, x):
        """
        Simulate the transmission of the signal x through the channel.
        This method is automatically called when you apply the module to an input tensor.
        """
        innovation = torch.zeros(x.size())
        for i in range(x.size(1)):
            innovation[:, i] = torch.normal(self.channel_noise_mean, np.sqrt(self.channel_noise_std / x.size(1)), size=[x.size(0)], dtype=x.dtype)
        innovation = innovation.to(x.device)
        # Creating the smooth noise with alpha
        if self.previous_noise is None:
            noise = innovation
        else:
            noise = self.alpha * self.previous_noise + innovation
        self.previous_noise = innovation.detach()   # in MA - Z_i = \alpha \epsilon_{i-1} + \epsilon_i

        return x + (noise).detach()

    def erase_states(self):
        self.previous_noise = None


class GAR1Channel(nn.Module):
    def __init__(self, args, logger=print):
        super().__init__()
        self.channel_info = args.channel_ndg_info
        self.channel_feedback = self.channel_info['channel_feedback']
        self.channel_noise_mean = 0
        self.channel_noise_std = self.channel_info['sigma_noise']
        self.alpha = self.channel_info['alpha']
        self.p_constraint = torch.tensor(args.channel_ndg_info['ndg']['constraint_value'])  # power constraint of NDG
        if args.x_dim == args.y_dim:
            self.dim = args.y_dim
            self.capacity_gt = self.calculate_capacity(self.p_constraint, self.channel_noise_std, self.alpha, feedback=self.channel_feedback)
            if self.capacity_gt is None:
                del self.capacity_gt
        self.previous_noise = None
        logger.info('| GAR(1) channel with dimension {} and SNR {} dB (at maximum capacity!)'.format(self.dim,
                    round(10 * np.log10(self.p_constraint.item() / self.channel_noise_std), 2)))

    @staticmethod
    def water_filling(noise_frequency_power_vec, p_condition, tol):
        K = len(noise_frequency_power_vec)  # K is the number of frequency slices

        # Initial step
        water_line = np.min(noise_frequency_power_vec) + p_condition / K  # Initialize waterline
        omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
        initial_P_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
        ptot = (1 / (2 * np.pi)) * np.trapz(initial_P_vec, omega_vec)  # Calculate total power for current waterline

        # Iteratively perform water filling
        while abs(p_condition - ptot) > tol:  # Continue as long as the error is higher than the given tolerance
            water_line = water_line + (p_condition - ptot) / K  # Raise water line
            p_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
            ptot = (1 / (2 * np.pi)) * np.trapz(p_vec, omega_vec)  # Compute new total power

        return water_line

    def calculate_capacity(self, P, N, alpha, feedback=True):
        if feedback:
            # https://arxiv.org/pdf/cs/0602091.pdf
            P = P / N  # to adjust P when N is not 1
            coeff = [P * np.abs(alpha) ** 2,
                     2 * P * np.abs(alpha),
                     P + 1,
                     0,
                     -1]
            roots = np.roots(coeff)
            x0 = roots[np.all([roots.real >= 0, np.isreal(roots)], axis=0)].real[0]
            capacity = -np.log(x0)
        else:
            P = P/N         # to adjust P when N is not 1
            K = 10000
            tolerance = 1e-6
            # Get K slices of the PSD of Z_i
            omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
            denominator = 1 + alpha * np.exp(-1j * omega_vec)
            H_w = np.abs(1 / denominator) ** 2
            # Calculate water line
            water_line = self.water_filling(H_w, P.item(), tolerance)
            # Calculate power allocation for each water line outcome
            P_allocation = np.maximum(0, water_line - H_w)
            # Compute allocation error (sanity check)
            allocation_error = np.sum(P_allocation - P.item())
            # Calculate capacity for the single value of P
            ones_vec = np.ones(K)
            int_vec = 0.5 * np.log(ones_vec + (1/self.dim) * P_allocation / H_w)
            capacity = self.dim * (1 / (2 * np.pi)) * np.trapz(int_vec, omega_vec)
        return capacity

    def forward(self, x):
        """
        Simulate the transmission of the signal x through the channel.
        This method is automatically called when you apply the module to an input tensor.
        """
        innovation = torch.zeros(x.size())
        for i in range(x.size(1)):
            innovation[:, i] = torch.normal(self.channel_noise_mean, np.sqrt(self.channel_noise_std / x.size(1)), size=[x.size(0)], dtype=x.dtype)
        innovation = innovation.to(x.device)
        # Creating the smooth noise with alpha
        if self.previous_noise is None:
            noise = innovation
        else:
            noise = self.alpha * self.previous_noise + innovation
        self.previous_noise = noise.detach()    # in AR - Z_i = \alpha Z_{i-1} + \epsilon_i

        return x + (noise).detach()

    def erase_states(self):
        self.previous_noise = None


class GMA100Channel(nn.Module):
    def __init__(self, args, logger=print):
        super().__init__()
        self.channel_info = args.channel_ndg_info
        self.channel_feedback = self.channel_info['channel_feedback']
        self.channel_noise_mean = 0
        self.channel_noise_std = self.channel_info['sigma_noise']
        self.alpha = self.channel_info['alpha']
        self.p_constraint = torch.tensor(args.channel_ndg_info['ndg']['constraint_value'])  # power constraint of NDG
        if args.x_dim == args.y_dim:
            self.dim = args.y_dim
            self.capacity_gt = self.calculate_capacity(self.p_constraint, self.channel_noise_std, self.alpha, feedback=self.channel_feedback)
            if self.capacity_gt is None:
                del self.capacity_gt
        self.previous_noise = [None] * 120
        logger.info('| Note! This channel has memory of 100 steps. It is recommended to use it with a sequence length of at least 100. ll={}, sl={}, pl={}'.format(args.label_len, args.seq_len, args.pred_len))
        logger.info('| GMA(100) channel with dimension {} and SNR {} dB (at maximum capacity!)'.format(self.dim,
                    round(10 * np.log10(self.p_constraint.item() / self.channel_noise_std), 2)))

    @staticmethod
    def water_filling(noise_frequency_power_vec, p_condition, tol):
        K = len(noise_frequency_power_vec)  # K is the number of frequency slices

        # Initial step
        water_line = np.min(noise_frequency_power_vec) + p_condition / K  # Initialize waterline
        omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
        initial_P_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
        ptot = (1 / (2 * np.pi)) * np.trapz(initial_P_vec, omega_vec)  # Calculate total power for current waterline

        # Iteratively perform water filling
        while abs(p_condition - ptot) > tol:  # Continue as long as the error is higher than the given tolerance
            water_line = water_line + (p_condition - ptot) / K  # Raise water line
            p_vec = np.maximum(water_line - noise_frequency_power_vec, 0)
            ptot = (1 / (2 * np.pi)) * np.trapz(p_vec, omega_vec)  # Compute new total power

        return water_line

    def calculate_capacity(self, P, N, alpha, feedback=True):
        if feedback:
            # https://arxiv.org/pdf/cs/0411036.pdf
            P = P / N  # to adjust P when N is not 1
            coeff = [-np.abs(alpha) ** 2,
                     2 * np.abs(alpha),
                     np.abs(alpha) ** 2 - 1 - P,
                     -2 * np.abs(alpha),
                     1]
            roots = np.roots(coeff)
            x0 = roots[np.all([roots.real >= 0, np.isreal(roots)], axis=0)].real[0]
            capacity = -np.log(x0)
        else:
            P = P/N         # to adjust P when N is not 1
            K = 10000
            tolerance = 1e-6
            # Get K slices of the PSD of Z_i
            omega_vec = np.linspace(-np.pi, np.pi - 1 / K, K)
            H_w = 1 + alpha ** 2 + 2 * alpha * np.cos(omega_vec)
            # Calculate water line
            water_line = self.water_filling(H_w, P.item(), tolerance)
            # Calculate power allocation for each water line outcome
            P_allocation = np.maximum(0, water_line - H_w)
            # Compute allocation error (sanity check)
            allocation_error = np.sum(P_allocation - P.item())
            # Calculate capacity for the single value of P
            ones_vec = np.ones(K)
            int_vec = 0.5 * np.log(ones_vec + (1/self.dim) * P_allocation / H_w)
            capacity = self.dim * (1 / (2 * np.pi)) * np.trapz(int_vec, omega_vec)
        return capacity

    def forward(self, x):
        """
        Simulate the transmission of the signal x through the channel.
        This method is automatically called when you apply the module to an input tensor.
        """
        innovation = torch.zeros(x.size())
        for i in range(x.size(1)):
            innovation[:, i] = torch.normal(self.channel_noise_mean, np.sqrt(self.channel_noise_std / x.size(1)), size=[x.size(0)], dtype=x.dtype)
        innovation = innovation.to(x.device)
        # Creating the smooth noise with alpha
        if self.previous_noise[-100] is None:
            noise = innovation
        else:
            noise = self.alpha * self.previous_noise[-100] + innovation
        self.previous_noise.pop(0)  # remove the oldest noise (the first one)
        self.previous_noise.append(innovation.detach())   # in MA - Z_i = \alpha \epsilon_{i-100} + \epsilon_i

        return x + (noise).detach()

    def erase_states(self):
        self.previous_noise = [None] * 120


if __name__ == "__main__":
    # test
    channel = AWGNChannel()
    x = torch.randn(1, 2, 2)
    y = channel(x)
    print(x)
    print(y)
