import ipdb as pdb
from model import RNN
from utils import create_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse


parser = argparse.ArgumentParser(description='Choose learning type')
parser.add_argument('--classical', action='store_true', help='classical learning rules')
parser.add_argument('--bayesian', action='store_true', help='bayesian learning rules')


args = parser.parse_args()

if args.classical:
    learning_type = "classical"
elif args.bayesian:
    learning_type = "bayesian"
else:
    print("Please specify a learning type: --classical or --bayesian")
    exit()





# PARAMETERS
tau_target = 160
p = 0.1
g = 1.5
N = 500
tau_m = 10
lr = [.001]*500
V = np.zeros(N)
t = 0
epochs = 8000
t_grid = np.arange(0.0, epochs, 1)
update_every = 5
idx = 2
rnn = RNN(p, g, N, tau_m, tau_target)
func = create_function()




# to save time, a target function (and 10 inputs) has been generated and saved in a pkl file.
# to generate new data, uncomment the following lines and modify the params:
# target_function_time, target_function_values = func.sample_from_gp(t_grid, tau_target, 5)
# input_time, input_amplitude = rnn.sample_inputs(duration_ms=10, sampling_rate=1)
# #save target function, target function values, input time, input amplitude to one pickle and load it:
# with open('data.pkl', 'wb') as f:
#     pickle.dump([target_function_time, target_function_values, input_time, input_amplitude], f)





lrs = []
delta_var_is = []
var_deltas = []
var_is = []
for i in range(1):

    with open('data.pkl', 'rb') as f:
        target_function_time, target_function_values, input_time, input_amplitude = pickle.load(f)

    readout_samples = []
    mse_values = []
    for epoch in range(epochs):    
        V = rnn.forward_pass(epoch, input_time, input_amplitude)
        readout_samples.append(np.dot(rnn.w, np.tanh(rnn.V)))


        if (epoch % 160 == 0) and (epoch != 0):  # validation epochs
            mean_error = np.mean(((target_function_values[idx][epoch-160:epoch-1]) - (readout_samples[epoch-160:epoch-1])) ** 2)
            print('MSE at epoch {}:'.format(epoch), mean_error)
            mse_values.append(mean_error)

        
        elif (epoch % update_every == 0) and (epoch != 0):  # training epochs
            error = (target_function_values[idx][epoch-update_every:epoch-1]) - (readout_samples[epoch-update_every:epoch-1])
            var_delta = np.var(error)


            # initialize var_i so that var_i / var_delta = .01:
            #first time in this condition only:                
            if (epoch == update_every) and (i == 0):
                var_i = .01 * var_delta
            del_var_i = -(np.power(var_i, 2) / var_delta) * (np.power(np.tanh(rnn.V), 2))
            # pdb.set_trace()
            var_i = var_i + del_var_i
            if learning_type == "bayesian":
                lr = (var_i) / (var_delta)
            if learning_type == "classical":
                lr = [.01]*500

            lr = np.array(lr)




            lrs.append(lr[0])
            # pdb.set_trace()
            delta_var_is.append(del_var_i[0])
            var_deltas.append(var_delta)
            var_is.append(var_i[0])


            rnn.w = rnn.w + lr * sum(error) * np.tanh(rnn.V)





# uncomment to plot misc graphs:

# plt.figure()
# plt.plot(lrs[0])
# plt.savefig('lrs.png')

# plt.figure()
# plt.plot(delta_var_is)
# plt.savefig('delta_var_is.png')


# plt.figure()
# plt.plot(var_deltas)
# plt.savefig('var_deltas.png')

# plt.figure()
# plt.plot(var_is)
# plt.savefig('var_is.png')




# new code for plotting
plt.figure()
# make plot wider
plt.figure(figsize=(15, 5))
plt.xlabel('timesteps 0 - 1600')
plt.xlim(0, 160*10)
plt.plot(target_function_values[2], label='Target function')
plt.plot(readout_samples, label='Estimated function')
plt.title('RNN Tracking target function (classical learning rules)')
plt.legend()
plt.savefig('tracking.png')



plt.figure()
plt.plot(mse_values)   
plt.xlabel('Time (validations)') 
plt.ylabel('MSE')
plt.title('MSE over time')
plt.savefig('mse.png')