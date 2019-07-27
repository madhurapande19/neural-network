import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse as ap

def sigm(x):
    return 1/(1 + np.exp(-x))


# def tanh(x):
#     return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh(x):
    return 2*sigm(2*x)-1

def softmax(x):
    h = np.exp(x) / (np.exp(x).sum(axis=1)[:, None])
    return h

def stable_softmax(x):
    m = np.max(x, axis=1)
    x_new = x - np.reshape(m, (x.shape[0], 1))
    return np.exp(x_new) / (np.exp(x_new).sum(axis=1)[:, None])

def grad_sigmoid(x):
    return sigm(x)*(1-sigm(x))


def grad_tanh(x):
    return 1-tanh(x)**2


def grad_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def get_gradient(x, func='sig'):
    if func == 'sig':
        return grad_sigmoid(x)
    elif func == 'tanh':
        return grad_tanh(x)
    elif func == 'relu':
        return grad_relu(x)


def write_to_file(max_epochs, loss_vec_train, loss_vec_val, nn_per_layer): # nnpaerlayer
    filename_epochs = 'epochs'+str(nn_per_layer)+'.txt'
    filename_loss_train = 'loss_train'+str(nn_per_layer)+'.txt'
    filename_loss_val = 'loss_val'+str(nn_per_layer)+'.txt'
    np.savetxt(filename_epochs, np.arange(max_epochs), fmt='%i')
    np.savetxt(filename_loss_train, loss_vec_train)
    np.savetxt(filename_loss_val, loss_vec_val)


def save_weights(list_of_weights, epoch, save_dir):
      with open(save_dir +'/weights_{}.pkl'.format(epoch), 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights(state, save_dir):
      with open(save_dir +'/weights_{}.pkl'.format(state), 'rb') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights


class Layer:
    def __init__(self, curr_layer_size, prev_layer_size, isIdentityLayer):

        self.layer_size = curr_layer_size
        self.pre_activation_output = 0.
        self.post_activation_output = 0.
        self.gradient_wrt_pre_activation = np.zeros(curr_layer_size)
        self.gradient_wrt_post_activation = np.zeros(curr_layer_size)
        self.gradient_wrt_weights = np.zeros((curr_layer_size, prev_layer_size))
        self.gradient_wrt_bias = np.zeros(curr_layer_size)

        self.momentum_update_weight = np.zeros((curr_layer_size, prev_layer_size))
        self.momentum_update_bias = np.zeros(curr_layer_size)

        self.velocity_update_weight = np.zeros((curr_layer_size, prev_layer_size))
        self.velocity_update_bias = np.zeros(curr_layer_size)

        np.random.seed(1234)
        if isIdentityLayer:
            self.weight = np.eye(curr_layer_size)
            self.bias = np.zeros(curr_layer_size)  # todo: check here
        else:
            self.weight = (1/np.sqrt(prev_layer_size)) * np.random.randn(curr_layer_size, prev_layer_size)  # todo: change this aptly later
            self.bias = np.zeros(curr_layer_size)

    """
    Input: inputs-> (n, d)
    Output : (d,n)
    """
    def calc_fwd_output(self, inputs,  apply_bn=False, activation='sig'):
        a = self._calc_linear_output(inputs)
        h = self._apply_non_linearity(a, activation)
        self.output = h
        self.pre_activation_output = a
        self.post_activation_output = h
        if apply_bn:
            self.post_activation_output = self.batch_normalization(h)

    def _calc_linear_output(self, inputs):
        return (np.dot(self.weight, inputs.T) + np.reshape(self.bias, (self.layer_size, 1))).T

    def _apply_non_linearity(self, inputs, activation):
        if activation == 'sig':
            h = sigm(inputs)
        elif activation == 'relu':
            h = np.maximum(0, inputs)
        elif activation == 'softmax':
            #h = np.exp(inputs) / (np.exp(inputs).sum(axis=1)[:, None])
            h = stable_softmax(inputs)
        elif activation == 'tanh':
            h = tanh(inputs)
        return h

    def reset_gradients(self):
        self.gradient_wrt_pre_activation[:] = 0.
        self.gradient_wrt_post_activation[:] = 0.
        self.gradient_wrt_weights[::] = 0.
        self.gradient_wrt_bias[:] = 0.

    def batch_normalization(self, x):
        feature_mean = np.mean(x, axis=0)
        feature_std = np.std(x, axis=0)
        x = x - feature_mean
        x = x / feature_std
        return x


class NeuralNetwork:

    def __init__(self, num_hidden, neurons_per_hidden_layer, input_dimension, output_layer_size, non_linearity_type, loss_type, applyBN = False):  # ouput size is same as no. of classes

        # construct layer0 (input layer)
        l = Layer(input_dimension, 0, True)
        self.layers = np.array(l)

        prev_layer_size = input_dimension

        for i in range(0, num_hidden):
            l = Layer(neurons_per_hidden_layer[i], prev_layer_size, False)
            prev_layer_size = neurons_per_hidden_layer[i]
            self.layers = np.append(self.layers, l)

        output_layer = Layer(output_layer_size, prev_layer_size, False)
        self.layers = np.append(self.layers, output_layer)
        self.num_layers = len(self.layers)  # should be num_hidden+2
        self.apply_batch_normalization = applyBN
        self.non_linearity = non_linearity_type
        self.loss_type = loss_type

    def show_network(self, showgradinfo=False):
        print('Presenting to you a Neural network:')
        print('Input layer (Layer(0)) size:')
        print(self.layers[0].layer_size)
        for i in range(1, self.num_layers):
            print('Layer' + str(i))
            print('Layer size=' + str(self.layers[i].layer_size))
            print('Weight matrix dim=' + str(self.layers[i].weight.shape))
            print('Bias vector dim=' + str(self.layers[i].bias.shape))
            if showgradinfo:
                print(self.layers[i].gradient_wrt_pre_activation)
                print(self.layers[i].gradient_wrt_post_activation)
                print(self.layers[i].gradient_wrt_weights)
                print(self.layers[i].gradient_wrt_bias)

    """
    :input x Processes a batch x((n,d)), examples arranged row-wise
    :returns a,h matrices as output
    """
    def forward_pass(self, x):
        self.layers[0].pre_activation_output = np.array((x))
        self.layers[0].post_activation_output = np.array((x))
        self.layers[0].output = np.array((x))

        for i in range(1, self.num_layers - 1):
            self.layers[i].calc_fwd_output(x, self.apply_batch_normalization, self.non_linearity)
            x = self.layers[i].output
        self.layers[i + 1].calc_fwd_output(x, False, 'softmax')

    # one-hot vec ll be a row vector , output for all training samples arranged row wise
    def backward_pass_vec(self, y_one_hot):
        if self.loss_type == 'ce':
            self.layers[self.num_layers - 1].gradient_wrt_pre_activation = -(y_one_hot - self.layers[self.num_layers-1].output)
            delta = self.layers[self.num_layers - 1].gradient_wrt_pre_activation
        elif self.loss_type == 'sq':
            y_diff = self.layers[self.num_layers -1].output - y_one_hot
            y_diff_mul = y_diff * self.layers[self.num_layers -1].output
            self.layers[self.num_layers - 1].gradient_wrt_pre_activation = np.zeros((y_one_hot.shape[0], y_one_hot.shape[1]))
            for j in range(y_one_hot.shape[1]):
                identity = np.zeros((y_one_hot.shape[0], y_one_hot.shape[1]))
                identity[:, j]=1
                self.layers[self.num_layers - 1].gradient_wrt_pre_activation[:, j] = np.sum(y_diff_mul*(identity - self.layers[self.num_layers -1].output), axis=1)
            delta = self.layers[self.num_layers - 1].gradient_wrt_pre_activation

        for k in range(self.num_layers - 1, 0, -1):

            self.layers[k].gradient_wrt_weights = np.dot(delta.T, self.layers[k - 1].post_activation_output)
            self.layers[k].gradient_wrt_bias = np.sum(delta, axis=0)

            self.layers[k - 1].gradient_wrt_pre_activation = np.dot(delta, self.layers[k].weight) * get_gradient(self.layers[k-1].pre_activation_output, self.non_linearity)
            delta = self.layers[k - 1].gradient_wrt_pre_activation

    def reset_gradients_after_minibatch(self):
        for k in range(1, self.num_layers):
            self.layers[k].reset_gradients()

    def update_params(self, eta, minibatch_size):
        for k in range(1, self.num_layers):
            self.layers[k].weight -= (1 / minibatch_size) * eta * self.layers[k].gradient_wrt_weights
            self.layers[k].bias -= (1 / minibatch_size) * eta * self.layers[k].gradient_wrt_bias

    def momentum_update_params(self, neural_network_history, history_param, eta, minibatch_size):
        for k in range(1, self.num_layers):
            k_layer_weight_prev_update = neural_network_history.layers[k].weight
            k_layer_bias_prev_update = neural_network_history.layers[k].bias
            k_layer_weight_update = np.multiply(history_param, k_layer_weight_prev_update) + eta * self.layers[
                k].gradient_wrt_weights
            k_layer_bias_update = np.multiply(history_param, k_layer_bias_prev_update) + eta * self.layers[
                k].gradient_wrt_bias
            neural_network_history.layers[k].weight = k_layer_weight_update
            neural_network_history.layers[k].bias = k_layer_bias_update
            self.layers[k].weight -= (1 / minibatch_size) * k_layer_weight_update
            self.layers[k].bias -= (1 / minibatch_size) * k_layer_bias_update

    def nesterov_update_params(self, neural_network_history, neural_network_temp, history_param, eta, minibatch_size):
        for k in range(1, self.num_layers):
            k_layer_weight_prev_update = neural_network_history.layers[k].weight
            k_layer_bias_prev_update = neural_network_history.layers[k].bias
            k_layer_weight_update = np.multiply(history_param, k_layer_weight_prev_update) + eta * self.layers[
                k].gradient_wrt_weights
            k_layer_bias_update = np.multiply(history_param, k_layer_bias_prev_update) + eta * self.layers[
                k].gradient_wrt_bias
            neural_network_history.layers[k].weight = k_layer_weight_update
            neural_network_history.layers[k].bias = k_layer_bias_update
            w_t = neural_network_temp.layers[k].weight
            b_t = neural_network_temp.layers[k].bias
            self.layers[k].weight = w_t - ((1 / minibatch_size) * k_layer_weight_update)
            self.layers[k].bias = b_t - ((1 / minibatch_size) * k_layer_bias_update)

    def adam_update_params(self, iteration_no, minibatch_size, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for k in range(1, self.num_layers):

            momentum_update = beta1*self.layers[k].momentum_update_weight + (1-beta1)*self.layers[k].gradient_wrt_weights
            self.layers[k].momentum_update_weight = momentum_update

            velocity_update = beta2*self.layers[k].velocity_update_weight + (1-beta2)*(self.layers[k].gradient_wrt_weights)**2
            self.layers[k].velocity_update_weight = velocity_update

            momentum_update_hat = momentum_update/(1-beta1**iteration_no)
            velocity_update_hat = velocity_update/(1-beta2**iteration_no)

            self.layers[k].weight -= (1 / minibatch_size) * ((eta * momentum_update_hat)/np.sqrt(velocity_update_hat+epsilon))

            momentum_update = beta1 * self.layers[k].momentum_update_bias + (1 - beta1) * self.layers[
                k].gradient_wrt_bias
            self.layers[k].momentum_update_bias = momentum_update

            velocity_update = beta2 * self.layers[k].velocity_update_bias + (1 - beta2) * self.layers[
                k].gradient_wrt_bias ** 2
            self.layers[k].velocity_update_bias = velocity_update

            momentum_update_hat = momentum_update / (1 - (beta1 ** iteration_no))
            velocity_update_hat = velocity_update / (1 - (beta2 ** iteration_no))
            self.layers[k].bias -= (1 / minibatch_size) * ((eta * momentum_update_hat)/np.sqrt(velocity_update_hat+epsilon))

    def adam_update_params_tf(self, iteration_no, minibatch_size, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):

        for k in range(1, self.num_layers):
            eta = eta* (np.sqrt(1-beta2**iteration_no)/(1-beta1**iteration_no))
            momentum_update = beta1*self.layers[k].momentum_update_weight + (1-beta1)*self.layers[k].gradient_wrt_weights
            self.layers[k].momentum_update_weight = momentum_update

            velocity_update = beta2*self.layers[k].velocity_update_weight + (1-beta2)*(self.layers[k].gradient_wrt_weights)**2
            self.layers[k].velocity_update_weight = velocity_update
            self.layers[k].weight -= (1 / minibatch_size) * ((eta * momentum_update)/(np.sqrt(velocity_update)+epsilon))

            momentum_update = beta1 * self.layers[k].momentum_update_bias + (1 - beta1) * self.layers[
                k].gradient_wrt_bias
            self.layers[k].momentum_update_bias = momentum_update

            velocity_update = beta2 * self.layers[k].velocity_update_bias + (1 - beta2) * self.layers[
                k].gradient_wrt_bias ** 2
            self.layers[k].velocity_update_bias = velocity_update
            self.layers[k].bias -= (1 / minibatch_size) * ((eta * momentum_update)/(np.sqrt(velocity_update)+epsilon))



    def predict(self, x_test, no_of_classes):
        y_pred = np.array([])
        y_hat_all = np.empty((0, no_of_classes))
        self.forward_pass(x_test)
        y_hat_all = self.layers[self.num_layers-1].output
        y_pred = np.argmax(y_hat_all, axis=1)
        return y_pred, y_hat_all

    def calc_accuracy(self, y_pred, y):
        y_diff = y - y_pred
        right_preds = np.count_nonzero(y_diff == 0)
        accuracy = (right_preds / len(y_diff)) * 100
        return accuracy

    # can this be optmized, remove loop??
    def calc_loss(self, y_hat_all, y):
        loss = 0.
        m = len(y)
        if self.loss_type == 'ce':
            for i in range(0, len(y)):
                l = int(y[i])
                loss -= np.log(y_hat_all[i, l])
        elif self.loss_type == 'sq':
            one_hot_vec = np.zeros((y_hat_all.shape[0], y_hat_all.shape[1]))
            one_hot_vec[np.arange(y_hat_all.shape[0]), y.astype(int)] = 1
            per_sample_loss = np.sum((one_hot_vec - y_hat_all)**2, axis=1)
            loss = np.sum(per_sample_loss)

        return (1. / m) * loss

    def calc_error(self, y_pred, y ):
        num = len(y_pred)
        y_diff = y - y_pred
        right_preds = np.count_nonzero(y_diff == 0)
        wrong_preds = num - right_preds
        error = (wrong_preds / num)*100
        return error


class NeuralNetworkUtil:

    def __init__(self, eta):
        self.learning_rate = eta # best prev value 0.01

    def momentum_gd(self, neural_network, neural_network_history, x_train, y_train, x_val, y_val, no_of_classes, max_epochs, save_dir, history_param=0.5,
                    minibatch_size=1):
        t = 0
        loss_vec_train = np.array([])
        loss_vec_val = np.array([])
        while t < max_epochs:
            random_index = np.random.permutation(x_train.shape[0])
            X_train, Y_train = x_train[random_index, :], y_train[random_index]
            for i in range(0, X_train.shape[0], minibatch_size):
                mini_data, mini_data_y = X_train[i:i + minibatch_size, :], Y_train[i:i + minibatch_size]
                neural_network.forward_pass(mini_data)
                one_hot_vec_for_minibatch = np.zeros((mini_data.shape[0], no_of_classes))
                one_hot_vec_for_minibatch[np.arange(mini_data.shape[0]), mini_data_y.astype(int)] = 1
                neural_network.backward_pass_vec(one_hot_vec_for_minibatch)
                if t >= 20:
                    self.learning_rate = 0.4
                neural_network.momentum_update_params(neural_network_history, history_param, self.learning_rate,
                                                      minibatch_size)
                neural_network.reset_gradients_after_minibatch()
                step_no = i/minibatch_size+1
                if (step_no % 100) == 0:
                    y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_train[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'train')

                    y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_val[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'val')

            if t % 1 == 0:
                y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                loss_vec_train = np.append(loss_vec_train, loss)
                print('Loss after epoch{} is {}'.format(t, loss))

                y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_val[0:1000])
                loss_vec_val = np.append(loss_vec_val, loss)
            list_weights = []
            for i in range(1, neural_network.num_layers):
                list_weights.append(neural_network.layers[i].weight)
                list_weights.append(neural_network.layers[i].bias)

            save_weights(list_weights, t, save_dir)
            t += 1
        return loss_vec_train, loss_vec_val

    def gradient_descent_vec(self, neural_network, x_train, y_train, no_of_classes, max_iterations, anneal, x_val, y_val, save_dir, minibatch_size=1):
        t = 0
        #self.learning_rate = eta
        val_loss_old = -1
        loss_vec_train = np.array([])
        loss_vec_val = np.array([])
        error__vec_train = np.array([])
        error__vec_val = np.array([])
        while t < max_iterations:
            idx = np.random.permutation(x_train.shape[0])
            X_train, Y_train = x_train[idx, :], y_train[idx]
            for i in range(0, X_train.shape[0], minibatch_size):
                mini_data, mini_data_y = X_train[i:i + minibatch_size, :], Y_train[i:i + minibatch_size]
                neural_network.forward_pass(mini_data)
                one_hot_vec_for_minibatch = np.zeros((mini_data.shape[0], no_of_classes))
                one_hot_vec_for_minibatch[np.arange(mini_data.shape[0]), mini_data_y.astype(int)] = 1

                neural_network.backward_pass_vec(one_hot_vec_for_minibatch)
                neural_network.update_params(self.learning_rate, minibatch_size)
                neural_network.reset_gradients_after_minibatch()
                step_no = i / minibatch_size + 1
                if (step_no % 100) == 0:
                    y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_train[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'train')

                    y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_val[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'val')

            if t % 1 == 0:
                y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                loss_vec_train = np.append(loss_vec_train, loss)
                print('Loss after epoch{} is {}'.format(t, loss))

                y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_val[0:1000])
                loss_vec_val = np.append(loss_vec_val, loss)

                if anneal:
                    y_pred_val, y_hat_all_val = neural_network.predict(x_val, 10)
                    val_loss_new = neural_network.calc_loss(y_hat_all_val, y_val)
                    if val_loss_new < val_loss_old:
                        self.learning_rate = 0.5 * self.learning_rate
                        t -= 1
                        val_loss_old = val_loss_new
            list_weights = []
            for i in range(1, neural_network.num_layers):
                list_weights.append(neural_network.layers[i].weight)
                list_weights.append(neural_network.layers[i].bias)

            save_weights(list_weights, t, save_dir)
            t += 1

        return loss_vec_train, loss_vec_val

    def nesterov_gd(self, neural_network, neural_network_history, neural_network_temp, x_train, y_train, x_val, y_val, no_of_classes,
                    max_epochs, save_dir, history_param=0.5, minibatch_size=1):
        t = 0
        loss_vec_train = np.array([])
        loss_vec_val = np.array([])
        while t < max_epochs:
            random_index = np.random.permutation(x_train.shape[0])
            X_train, Y_train = x_train[random_index, :], y_train[random_index]
            for i in range(0, X_train.shape[0], minibatch_size):
                mini_data, mini_data_y = X_train[i:i + minibatch_size, :], Y_train[i:i + minibatch_size]
                neural_network.forward_pass(mini_data)

                # lookahead logic
                for k in range(1, neural_network.num_layers, 1):
                    neural_network_temp.layers[k].weight = neural_network.layers[k].weight
                    neural_network_temp.layers[k].bias = neural_network.layers[k].bias
                    neural_network.layers[k].weight = neural_network_temp.layers[k].weight - np.multiply(history_param,
                                                                                                         neural_network_history.layers[
                                                                                                             k].weight)
                    neural_network.layers[k].bias = neural_network_temp.layers[k].bias - np.multiply(history_param,
                                                                                                     neural_network_history.layers[
                                                                                                         k].bias)

                one_hot_vec_for_minibatch = np.zeros((mini_data.shape[0], no_of_classes))
                one_hot_vec_for_minibatch[np.arange(mini_data.shape[0]), mini_data_y.astype(int)] = 1
                neural_network.backward_pass_vec(one_hot_vec_for_minibatch)
                if t >= 10:
                    self.learning_rate = 0.4
                neural_network.nesterov_update_params(neural_network_history, neural_network_temp, history_param,
                                                      self.learning_rate,
                                                      minibatch_size)
                neural_network.reset_gradients_after_minibatch()
                step_no = i / minibatch_size + 1
                if (step_no % 100) == 0:
                    y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_train[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'train')

                    y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_val[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'val')
            if t % 1 == 0:
                y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                loss_vec_train = np.append(loss_vec_train, loss)
                print('Loss after epoch{} is {}'.format(t, loss))

                y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_val[0:1000])
                loss_vec_val = np.append(loss_vec_val, loss)
            list_weights = []
            for i in range(1, neural_network.num_layers):
                list_weights.append(neural_network.layers[i].weight)
                list_weights.append(neural_network.layers[i].bias)

            save_weights(list_weights, t, save_dir)
            t += 1

        return loss_vec_train, loss_vec_val

    def adam(self, neural_network, x_train, y_train, no_of_classes, max_iterations, x_val, y_val, save_dir, minibatch_size=20):
        #eta = 0.01
        t = 0
        loss_vec_train = np.array([])
        loss_vec_val = np.array([])
        while t < max_iterations:
            idx = np.random.permutation(x_train.shape[0])
            X_train, Y_train = x_train[idx, :], y_train[idx]
            iteration_no = 1
            for i in range(0, X_train.shape[0], minibatch_size):

                mini_data, mini_data_y = X_train[i:i + minibatch_size, :], Y_train[i:i + minibatch_size]

                neural_network.forward_pass(mini_data)
                one_hot_vec_for_minibatch = np.zeros((mini_data.shape[0], no_of_classes))
                one_hot_vec_for_minibatch[np.arange(mini_data.shape[0]), mini_data_y.astype(int)] = 1
                neural_network.backward_pass_vec(one_hot_vec_for_minibatch)

                neural_network.adam_update_params(iteration_no, minibatch_size, self.learning_rate)
                if iteration_no % minibatch_size == 0:
                    iteration_no = 1
                else:
                    iteration_no += 1
                step_no = i / minibatch_size + 1
                if (step_no % 100) == 0:
                    y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_train[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'train')

                    y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                    error = neural_network.calc_error(y_pred[0:1000], y_val[0:1000])
                    loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                    self.create_log_entry(t, step_no, loss, error, self.learning_rate, 'val')

            if t % 1 == 0:
                y_pred, y_hat_all = neural_network.predict(x_train[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_train[0:1000])
                loss_vec_train = np.append(loss_vec_train, loss)
                print('Loss after epoch{} is {}'.format(t, loss))

                y_pred, y_hat_all = neural_network.predict(x_val[0:1000, :], no_of_classes)
                loss = neural_network.calc_loss(y_hat_all[0:1000, :], y_val[0:1000])
                loss_vec_val = np.append(loss_vec_val, loss)

            list_weights = []
            for i in range(1, neural_network.num_layers):
                list_weights.append(neural_network.layers[i].weight)
                list_weights.append(neural_network.layers[i].bias)

            save_weights(list_weights, t, save_dir)
            t+=1
        return loss_vec_train, loss_vec_val

    def train(self, neural_network, x_train, y_train, no_of_classes, anneal, x_val, y_val, nn_per_layer, opt, momentum_history_param, minibatch_size, max_epochs, save_dir):

        #minibatch_size = 20

        if opt == 'gd':
            print('Running gradient descent now')
            #max_epochs = 1
            loss_vec_train, loss_vec_val = util.gradient_descent_vec(neural_network, x_train, y_train, no_of_classes, max_epochs, anneal, x_val, y_val, save_dir, minibatch_size)
            #write_to_file(max_epochs, loss_vec_train, loss_vec_val, nn_per_layer)

        if opt == 'momentum':
            print('Running momentum now')
            #max_epochs = 3
            #momentum_history_param = 0.2
            neural_network_history = NeuralNetwork(num_hidden, sizes, x_train.shape[1], 10, neural_network.non_linearity, neural_network.loss_type, False)
            for k in range(1, neural_network_history.num_layers, 1):
                 current_layer_size = neural_network_history.layers[k].weight.shape[0]
                 prev_layer_size = neural_network_history.layers[k].weight.shape[1]
                 neural_network_history.layers[k].weight = np.zeros((current_layer_size, prev_layer_size))
                 neural_network_history.layers[k].bias = np.zeros(current_layer_size)

            loss_vec_train, loss_vec_val = util.momentum_gd(neural_network, neural_network_history, x_train, y_train, x_val, y_val, no_of_classes, max_epochs, save_dir, momentum_history_param, minibatch_size)
            #write_to_file(max_epochs, loss_vec_train, loss_vec_val, nn_per_layer)

        if opt == 'nag':
            print('Running Nesterov accelerated gradient descent now')
            #max_epochs = 2
            momentum_history_param = 0.1
            neural_network_history = NeuralNetwork(num_hidden, sizes, x_train.shape[1], 10, neural_network.non_linearity, neural_network.loss_type, False)
            neural_network_temp = NeuralNetwork(num_hidden, sizes, x_train.shape[1], 10, neural_network.non_linearity, neural_network.loss_type, False)
            for k in range(1, neural_network_history.num_layers, 1):
                current_layer_size = neural_network_history.layers[k].weight.shape[0]
                prev_layer_size = neural_network_history.layers[k].weight.shape[1]
                neural_network_history.layers[k].weight = np.zeros((current_layer_size, prev_layer_size))
                neural_network_history.layers[k].bias = np.zeros(current_layer_size)
                neural_network_temp.layers[k].weight = np.zeros((current_layer_size, prev_layer_size))
                neural_network_temp.layers[k].bias = np.zeros(current_layer_size)
            #neural_network_history.show_network()

            loss_vec_train, loss_vec_val = util.nesterov_gd(neural_network, neural_network_history, neural_network_temp, x_train, y_train, x_val, y_val, 10,
                                      max_epochs, save_dir, momentum_history_param, minibatch_size)
            #write_to_file(max_epochs, loss_vec_train, loss_vec_val, nn_per_layer)

        if opt == 'adam':
            print('Running Adam now')
            #max_epochs = 5
            loss_vec_train, loss_vec_val = util.adam(neural_network, x_train, y_train, no_of_classes, max_epochs, x_val, y_val, save_dir, minibatch_size)
            #write_to_file(max_epochs, loss_vec_train, loss_vec_val, nn_per_layer)

    def create_log_entry(self, epoch_no, step_no, loss, error, lr, type):

        filename = expt_dir+'/log_'

        if type == 'train':
            filename += 'train.txt'
        elif type == 'val':
            filename += 'val.txt'

        f = open(filename, 'a')
        error = '%.2f' % error
        loss = '%.2f' % loss
        line = 'Epoch ' +str(epoch_no)+', Step '+str(int(step_no))+', Loss: '+str(loss)+', Error: '+str(error)+', lr: '+str(lr)
        f.write(line+'\n')

#python train.py --pretrain True --state X --testing True --save_dir save_dir/best --expt_dir expt_dir



parser = ap.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--momentum")
parser.add_argument("--num_hidden")
parser.add_argument("--sizes")
parser.add_argument("--activation")
parser.add_argument("--loss")
parser.add_argument("--opt")
parser.add_argument("--batch_size")
parser.add_argument("--epochs")
parser.add_argument("--anneal")
parser.add_argument("--save_dir")
parser.add_argument("--expt_dir")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--test")
parser.add_argument("--pretrain")
parser.add_argument("--state")
parser.add_argument("--testing")

args = parser.parse_args()
testing = True if str(args.testing).lower() == 'true' else False
pretrain = True if str(args.pretrain).lower() == 'true' else False
state = args.state

if not testing:
    eta = float(args.lr)
    momentum_history_param = float(args.momentum)
    loss = str(args.loss).lower()
    opt = str(args.opt).lower()
    minibatch_size = int(args.batch_size)
    epochs = int(args.epochs)
    anneal = True if str(args.anneal).lower() == 'true' else False
    train = str(args.train)
    val = str(args.val)

num_hidden = int(args.num_hidden)
sizes = list(args.sizes.split(","))
sizes = list(map(int, sizes))
activation = 'sig' if str(args.activation).lower() == 'sigmoid' else 'tanh'
save_dir = str(args.save_dir)
expt_dir = str(args.expt_dir)
test = str(args.test)
no_of_classes = 10

if testing:
    print('Using pre-trained models for predictions')
    filename = save_dir+"/weights_"+state+".pkl"
    f = open(filename, 'rb')
    param_dict = pickle.load(f)
    f.close()
    loss = 'ce'
    weight_l1 = param_dict["Layer1"]["weight"]
    weight_l2 = param_dict["Layer2"]["weight"]
    weight_l3 = param_dict["Layer3"]["weight"]
    bias_l1 = param_dict["Layer1"]["bias"]
    bias_l2 = param_dict["Layer2"]["bias"]
    bias_l3 = param_dict["Layer3"]["bias"]
    neural_network = NeuralNetwork(num_hidden, sizes, weight_l1.shape[1], no_of_classes, activation, loss, False)
    neural_network.layers[1].weight = weight_l1
    neural_network.layers[2].weight = weight_l2
    neural_network.layers[3].weight = weight_l3

    neural_network.layers[1].bias = bias_l1
    neural_network.layers[2].bias = bias_l2
    neural_network.layers[3].bias = bias_l3

    filename = test
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1] - 1  # First col is id, last is class label
    id1 = data[:, 0]
    x_test = data[:, 1:feature_size + 1]
    feature_mean_test = np.mean(x_test, axis=0)
    feature_std_test = np.std(x_test, axis=0)
    x_test = x_test - feature_mean_test
    x_test = x_test / feature_std_test

    y_pred, y_hat_all = neural_network.predict(x_test, 10)
    final_output = np.concatenate((id1[:, None], y_pred[:, None]), axis=1)

    filename = expt_dir + "/predictions_" + state + ".csv"
    with open(filename, "wb") as f:
        f.write(b'id,label\n')
        np.savetxt(f, final_output.astype(int), fmt='%i', delimiter=",")


else:
    # Read data from file and start learning
    print('Reading data from file')
    filename = train
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-2  # First col is id, last is class label
    x_train = data[:, 1:feature_size+1]
    y_train = data[:, -1]

    # # Normalize features
    feature_mean = np.mean(x_train, axis=0)
    feature_std = np.std(x_train, axis=0)
    x_train = x_train - feature_mean
    x_train = x_train/feature_std

    filename = val
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-2  # First col is id, last is class label
    x_val = data[:, 1:feature_size+1]
    y_val = data[:, -1]

    feature_mean_val = np.mean(x_val, axis=0)
    feature_std_val = np.std(x_val, axis=0)
    x_val = x_val - feature_mean_val
    x_val = x_val/feature_std_val

    if pretrain:
        print('Using pre-trained weights from epoch_no {} to initialize'.format(state))
        filename = save_dir+"/weights_"+state+".pkl"
        f = open(filename,'rb')
        weight_list = pickle.load(f)
        f.close()
        #print(weight_list[0].shape[1])
        neural_network = NeuralNetwork(num_hidden, sizes, weight_list[0].shape[1], no_of_classes, activation, loss, False)
        weight_list = load_weights(state, save_dir)
        layer_no = 1
        for i in range(0, len(weight_list), 2):
            neural_network.layers[layer_no].weight = weight_list[i]
            neural_network.layers[layer_no].bias = weight_list[i+1]
            layer_no += 1

        epochs = epochs - int(state) - 1
        print('Pretrained till state {}'.format(state))
        print('Will run for remaining {} epochs'.format(epochs))
    else:
        neural_network = NeuralNetwork(num_hidden, sizes, x_train.shape[1], no_of_classes, activation, loss, False)

    neural_network.show_network()

    #Train the network
    util = NeuralNetworkUtil(eta)
    util.train(neural_network, x_train, y_train, no_of_classes, anneal, x_val, y_val, sizes[0], opt, momentum_history_param, minibatch_size, epochs, save_dir)
    y_pred, y_hat_all = neural_network.predict(x_train, no_of_classes)

    accuracy = neural_network.calc_accuracy(y_pred, y_train)
    print('Train accuracy' + str(accuracy))

    y_pred, y_hat_all = neural_network.predict(x_val, no_of_classes)
    accuracy = neural_network.calc_accuracy(y_pred, y_val.astype(int))
    print('Val accuracy' + str(accuracy))


    #For Test data
    filename = test
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-1  # First col is id, last is class label
    id1 = data[:, 0]
    x_test = data[:, 1:feature_size+1]
    feature_mean_test = np.mean(x_test, axis=0)
    feature_std_test = np.std(x_test, axis=0)
    x_test = x_test - feature_mean_test
    x_test = x_test/feature_std_test

    y_pred, y_hat_all = neural_network.predict(x_test, 10)
    final_output = np.concatenate((id1[:, None], y_pred[:, None]), axis=1)

    filename = expt_dir+"/predictions.csv"
    with open(filename, "wb") as f:
        f.write(b'id,label\n')
        np.savetxt(f, final_output.astype(int), fmt='%i', delimiter=",")

    # param_dict = {"Layer1": {"weight": neural_network.layers[1].weight, "bias": neural_network.layers[1].bias}, "Layer2": {"weight": neural_network.layers[2].weight, "bias": neural_network.layers[2].bias}, "Layer3": {"weight": neural_network.layers[3].weight, "bias": neural_network.layers[3].bias}}
    # filename = save_dir+"/weights_1_temp.pkl"
    # f = open(filename, 'wb')
    # pickle.dump(param_dict, f)
    #f.close()