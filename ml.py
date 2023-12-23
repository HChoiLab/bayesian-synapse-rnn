from model import RNN
from utils import create_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
# PARAMETERS
tau_target = 160
p = 0.1
g = 1.5
N = 500
tau_m = 10
lr = .001
V = np.zeros(N)
t = 0
epochs = 8000
t_grid = np.arange(0.0, epochs, 1)
update_every = 5
idx = 2
rnn = RNN(p, g, N, tau_m, tau_target)
func = create_function()


# target_function_time, target_function_values = func.sample_from_gp(t_grid, tau_target, 5)
# input_time, input_amplitude = rnn.sample_inputs(duration_ms=10, sampling_rate=1)

# #save target function, target function values, input time, input amplitude to one pickle and load it:
# with open('data.pkl', 'wb') as f:
#     pickle.dump([target_function_time, target_function_values, input_time, input_amplitude], f)

# #save target function, target function values, input time, input amplitude to one pickle and load it:     



# learning_type = "classical" or "bayesian"
learning_type = "bayesian"


for i in range(3):

    with open('data.pkl', 'rb') as f:
        target_function_time, target_function_values, input_time, input_amplitude = pickle.load(f)

    readout_samples = []
    mse_values = []
    for epoch in range(epochs):    
        V = rnn.forward_pass(epoch, input_time, input_amplitude)
        readout_samples.append(np.dot(rnn.w, rnn.V))
        
        if learning_type == "classical":
            if (epoch % 400 == 0) and (epoch != 0):  # validation epochs
                error = np.mean(((target_function_values[idx][epoch-100:epoch-1]) - (readout_samples[epoch-100:epoch-1])) ** 2)
                # error = np.mean((target_function_values[3][epoch] - readout_samples[epoch]) ** 2)
                # print('MSE at epoch {}:'.format(epoch), error)
                mse_values.append(error)

            elif (epoch % update_every == 0) and (epoch != 0):  # training epochs
                error = sum((target_function_values[idx][epoch-update_every:epoch-1] - readout_samples[epoch-update_every:epoch-1]))
                rnn.w = rnn.w + (lr * error * np.tanh(rnn.V))

        elif learning_type == "bayesian":
            if (epoch % 400 == 0) and (epoch != 0):  # validation epochs
                mean_error = np.mean(((target_function_values[idx][epoch-100:epoch-1]) - (readout_samples[epoch-100:epoch-1])) ** 2)
                # error = np.mean((target_function_values[3][epoch] - readout_samples[epoch]) ** 2)
                print('MSE at epoch {}:'.format(epoch), mean_error)
                mse_values.append(error)

            
            elif (epoch % update_every == 0) and (epoch != 0):  # training epochs
                error = (target_function_values[idx][epoch-update_every:epoch-1]) - (readout_samples[epoch-update_every:epoch-1])
                # update variance
                var_delta = np.var(error)

                # initialize var_i so that var_i / var_delta = .01:
                #first time in this condition only:                
                if (epoch == update_every) and (i == 0):
                    var_i = - math.sqrt(.01 * var_delta)
                del_var_i = -(np.power(var_i, 2) / var_delta) * np.tanh(rnn.V)
                var_i = var_i + del_var_i
                lr = (var_i) / (var_delta)
                # if (epoch == update_every) and (i == 0):
                #     print(lr)
                # print(lr.shape)
                lr = [.01]*500
                lr = np.array(lr)
                rnn.w = rnn.w + lr * sum(error) * np.tanh(rnn.V)






# new code for plotting
plt.figure()
# make plot wider
# plt.figure(figsize=(15, 5))
plt.xlim(5000, 6000)
plt.plot(target_function_values[2], label='Target function')
plt.plot(readout_samples, label='Estimated function')
plt.legend()
plt.savefig('outs.png')


plt.figure()
plt.plot(mse_values)    
plt.savefig('mse.png')
