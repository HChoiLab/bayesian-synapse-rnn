from model import RNN
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
tau_target = 400
p = 0.1
g = 1.5
N = 500
tau_m = 10
lr = .001
V = np.zeros(N)
t = 0
t_gp = np.linspace(0, 1000, 10000)  
epochs = 4000


rnn = RNN(p, g, N, tau_m, tau_target)



input_time, input_amplitude = rnn.sample_inputs(duration_ms=10, sampling_rate=1)
target_function_time, target_function_values = rnn.sample_from_gp(t_gp, tau_target, 5)







readout_samples = []
mse_values = []
for epoch in range(epochs):    
    V = rnn.forward_pass(epoch, input_time, input_amplitude)
    readout_samples.append(np.dot(rnn.w, rnn.V))
    
    # update the weights
    if (epoch % 400 == 0) and (epoch != 0):  # validation epochs
        error = np.mean(((target_function_values[3][epoch-100:epoch-1]) - (readout_samples[epoch-100:epoch-1])) ** 2)
        # error = np.mean((target_function_values[3][epoch] - readout_samples[epoch]) ** 2)
        print('MSE at epoch {}:'.format(epoch), error)
        mse_values.append(error)

    elif (epoch % 5 == 0) and (epoch != 0):  # training epochs
        error = sum((target_function_values[3][epoch-5:epoch-1] - readout_samples[epoch-5:epoch-1]))
        rnn.w = rnn.w + (lr * error * np.tanh(rnn.V))





# new code for plotting
plt.figure()
# plt.xlim(0, 1500)
plt.plot(target_function_values[3], label='Target function')
plt.plot(readout_samples, label='Estimated function')
plt.legend()
plt.savefig('outs.png')


plt.figure()
plt.plot(mse_values)    
plt.savefig('mse.png')
