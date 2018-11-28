import numpy as np
import pickle


class NeuralNet(object):
    """ 
    This class trains a feed-forward, fully connected neural net, using numpy as only dependency for calculation
    and pickle for saving models. 

    ACTIVATION FUNCTIONS: ReLU, Sigmoid
    OPTIMIZATION ALGORITHMS: Gradient Descent, ADAM
    REGULARIZATION: L2-regularization
    BATCH-NORMALIZATION: Yes (if turned on, it is implemented before every activation layer)

    Typical use:
        data = {...}                                # See definition in NeuralNet.load_data()
        model = [...]                               # See definition in NeuralNet.load_model()
        meta_params = {...}                         # See definition in NeuralNet.load_meta_params()
        nn = NeuralNet(meta_params, model, data)    # Set up the class
        nn.initialize_parameters()                  # Init model parameters
        nn.train(epochs=100)                        # Learn model parameters
        nn.clean_model(convert_batch_norm=True)     # Remove cached data from model (to reduce memory consumption)
        nn.evaluate('training')                     # Evaluate the training on the full training set
        nn.evaluate('test')                         # Evaluate the training on the full test set
        nn.evaluate('validation')                   # Evaluate the training on the full validation set
    """

    def __init__(self, meta_params=None, model=None, data=None):
        """
        :param meta_params: dict. See load_meta_params() for definition.
        :param model: dict. See load_model() for definition.
        :param data: dict. See load_data() for definition.
        """

        # Set default values
        self.meta_params = {
            'optim': {'algo': 'adam', 'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
                      'mini_batch_size': 128},
            'data_split': {'train': 1.0, 'test': 0.0, 'validation': 0.0},
            'regularization_lambda': 0.0,
            'batch_norm': {'use': False, 'epsilon': 1e-8},
            'print_cost_interval': 10
        }
        self.data_sets = None
        self.model = None
        self.costs = list()
        self.cost_counter = {'sum': 0.0, 'count': 0.0}

        # Load settings if provided
        if meta_params is not None:
            self.load_meta_params(meta_params)
        if data is not None:
            self.load_data(data)
        if model is not None:
            self.load_model(model)

    def load_data(self, data):
        """
        Loads and splits data in training, test and validation sets. The relative size of this split is read from 
        meta_params['data_split'], so the meta_params must be set before load_data is called. 
        Default split is meta_params['data_split'] = {'train': 1.0, 'test': 0.0, 'validation': 0.0}.

        data is a dict on the format below.

            data = {
                'x': my_data_x, # numpy.array with my_data_x.shape = (nb_of_input_variables, nb_of_observations)
                'y': my_data_y  # numpy.array with my_data_x.shape = (nb_of_output_variables, nb_of_observations)
            }

        :param data: dict. See description above for definition.
        :return: Nothing
        """

        # Verify that data is in correct format
        errors = ''
        if type(data) is not dict:
            errors += 'Error: Data must be a dict(). Data not loaded. \n'
        if 'x' not in data.keys() or 'y' not in data.keys():
            errors += 'Error: Data must be a dict() with x and y. \n'
        if data['x'].shape[1] != data['y'].shape[1]:
            errors += 'data["x"] an data["y"] must have the same number of columns. (Each column is an observation). \n'

        if len(errors) > 0:
            print(errors + 'WARNING: Data not loaded \n')
        else:
            # Permute data
            permutation = list(np.random.permutation(data['x'].shape[1]))
            x = data['x'][:, permutation].reshape((data['x'].shape[0], data['x'].shape[1]))
            y = data['y'][:, permutation].reshape((data['y'].shape[0], data['y'].shape[1]))

            # Cut data into training, test & validation sets
            cutoff_1 = round(x.shape[1] * self.meta_params['data_split']['train'])
            cutoff_2 = cutoff_1 + round(x.shape[1] * self.meta_params['data_split']['test'])
            data['x_training'] = x[:, :cutoff_1]
            data['y_training'] = y[:, :cutoff_1]
            data['x_test'] = x[:, cutoff_1:cutoff_2]
            data['y_test'] = y[:, cutoff_1:cutoff_2]
            data['x_validation'] = x[:, cutoff_2:]
            data['y_validation'] = y[:, cutoff_2:]
            self.data_sets = data

    def load_model(self, model):
        """
        Loads the model of the network. 

        model is a list of dicts, where each dict specifies a layer (or a group of layers). 
        The first layer must be an input layer. The last layer must be a cost layer. 
        No other cost layers can be included in the network. 

        SUPPORTED LAYERS
        Below is a list of supported layers (or groups of layers). 
        The item 'type' specifies which type of layer it is. 

        Input:
            'units' is the number of variables going out from the layer
            {'type': 'input', 'units': 4}   

        Linear + activation layer
            'units' is the number of variables going out from the layer
            'activation' is the activation function to use. Only 'relu' & 'sigmoid' are supported.
            {'type': 'lin_act', 'units': 1, 'activation': 'relu'}

        Cost layer:
            cost_function specifies which cost function should be used. Supported functions are: 
                'mean_squared_error' & 'cross-entropy'
            {'type': 'cost', 'cost_function': 'mean_squared_error'}

        Example:
            model = [
                {'type': 'input', 'units': 4}, 
                {'type': 'lin_act', 'units': 3, 'activation': 'relu'}, 
                {'type': 'lin_act', 'units': 2, 'activation': 'relu'}, 
                {'type': 'lin_act', 'units': 1, 'activation': 'relu'}, 
                {'type': 'cost', 'cost_function': 'mean_squared_error'}
            ]

        Note: the last layer before the cost layer must have a suitable number of output units (typically 1). 

        :param model: list of dicts, see definition above. 
        :return: Nothing. 
        """

        errors = ''
        if type(model) is not list:
            errors += 'Error: Model is not a list. Model must be a list of dicts. \n'
        for layer in model:
            if type(layer) is not dict:
                errors += 'Error: Model contains item that is not a dict. All items must be dicts. \n'
            else:
                if 'type' not in layer:
                    errors += 'Error: A layer did not contain required item "type". \n'
                if 'units' not in layer and 'type' in layer:
                    if layer['type'] == 'lin_act' or layer['type'] == 'input':
                        errors += 'Error: A layer did not contain required item "units". \n'
                if layer['type'] == 'lin_act' and 'activation' not in layer:
                    errors += 'Error: A lin_act layer did not contain required item "activation". \n'
        if len(errors) == 0:
            self.model = model
        else:
            print(errors + '\n WARNING: Model not loaded.')

    def load_meta_params(self, meta_params):
        """
        If meta_params is not specified, the following default values will be used: 
            model = {
                'optim': {'algo': 'adam', 'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
                          'mini_batch_size': 128},
                'data_split': {'train': 1.0, 'test': 0.0, 'validation': 0.0},
                'regularization_lambda': 0.0,
                'batch_norm': {'use': False, 'epsilon': 1e-8},
                'print_cost_interval': 10
            }

        The parameters can be changed, and you don't need to respecify the parameters that you leave as default.
        Some examples: 
            # Turn on Batch Normalization:
            nn.load_meta_params('batch_norm': {'use': True, 'epsilon': 1e-8}})
            # Turn on L2 Regularization:
            nn.load_meta_params({'regularization_lambda': 0.7})

        To set optimization algorithm to gradient descent with momentum:
            nn.load_meta_params({'optim': {'algo': 'gradient_descent', 'learning_rate': 0.01, 'beta': 0.9, 
                                'mini_batch_size': 128}})

        To set optimization algorithm to stochastic gradient descent, with momentum:
            nn.load_meta_params({'optim': {'algo': 'gradient_descent', 'learning_rate': 0.01, 'beta': 0.99, 
                                'mini_batch_size': 1}})
            nn.load_meta_params('batch_norm': {'use': False, 'epsilon': 1e-8}})  # Cannot use batch norm batch size = 1.


        Note: for values that are dicts (data_split, optim, batch_norm) the full dict must be given, as in the 
        batch-norm example above. 

        :param meta_params: dict on format as described above. 
        :return: Nothing
        """

        for key, value in meta_params.items():
            if key in self.meta_params:
                self.meta_params[key] = value
            else:
                print('Warning: "{0}" is not a valid meta parameter. It will be ignored.'.format(str(key)))

    def initialize_parameters(self, force_param_reset=False, force_optim_reset=False):
        """
        Initializes parameters that are used both in the model (W, b, bn_beta, bn_gamma) and in the optimization
        algorithm (v_W, v_b, s_W, s_b etc). 

        This should not be called until the model is loaded (to know parameter dimensions) and the meta_params are set 
        (as we need to know what optimization algorithm is used, and whether batch-norm is used). 

        By default parameters that already exist in the model are left unchanged, unless force_param_reset or 
        force_optim_reset are set to True. 

        :param force_param_reset: boolean. Defaults to False. 
        :param force_optim_reset: boolean. Defaults to False. 
        :return: Nothing. 
        """

        # Initializes W and b for each lin_act layer, where they are ar not initialized
        last_units = 0
        for layer in self.model:
            if layer['type'] == 'lin_act':
                units = layer['units']
                if 'b' not in layer or force_param_reset:
                    layer['b'] = np.zeros([units, 1]).astype(np.float32)
                if 'W' not in layer or force_param_reset:
                    if layer['activation'] == 'relu':
                        # He initialization
                        layer['W'] = np.random.randn(units, last_units).astype(np.float32) * np.sqrt(2.0 / (last_units))
                    else:
                        # Xavier initialization
                        layer['W'] = np.random.randn(units, last_units).astype(np.float32) * np.sqrt(1.0 / (last_units))
                last_units = units
            elif layer['type'] == 'input':
                last_units = layer['units']
                # elif layer['type'] == 'sigmoid':  # Do nothing
                # elif layer['type'] == 'softmax':  # Do nothing
                # elif layer['type'] == 'cost':     # Do nothing

        # Initialize parameters for batch norm, if used
        if self.meta_params['batch_norm']['use']:
            for layer in self.model:
                if layer['type'] == 'lin_act':
                    layer['bn_Z_mean'] = np.zeros(layer['b'].shape).astype(np.float32)
                    layer['bn_Z_var'] = np.ones(layer['b'].shape).astype(np.float32)
                    layer['bn_gamma'] = np.ones(layer['b'].shape).astype(np.float32)
                    layer['bn_beta'] = np.zeros(layer['b'].shape).astype(np.float32)
                    del layer['b']

        # Initialize optimization parameters, if ADAM is used
        if self.meta_params['optim']['algo'] == 'adam':
            self.meta_params['optim']['update_counter'] = 0.0
            for layer in self.model:
                if layer['type'] == 'lin_act':
                    if 'v_W' not in layer or force_optim_reset:
                        for param in [p for p in ['W', 'b', 'bn_gamma', 'bn_beta'] if p in layer]:
                            layer['v_' + param] = np.zeros(layer[param].shape)
                            layer['s_' + param] = np.zeros(layer[param].shape)

        # Initialize optimization parameters, if gradient descent is used
        if self.meta_params['optim']['algo'] == 'gradient_descent':
            for layer in self.model:
                if layer['type'] == 'lin_act':
                    if 'v_W' not in layer or force_optim_reset:
                        for param in [p for p in ['W', 'b', 'bn_gamma', 'bn_beta'] if p in layer]:
                            layer['v_' + param] = np.zeros(layer[param].shape)

    def train(self, epochs=10):
        print('Starting training, {} epochs...'.format(epochs))
        for e in range(epochs):
            self._reset_cost_counter()
            mini_batches = self._get_mini_batches()
            for mb in mini_batches:
                self._forward_prop(mb['x'], mb['y'], is_training=True)
                self._backward_prop(mb['x'], mb['y'])
                self._update_parameters()

            cost = self.cost_counter['sum'] / self.cost_counter['count']
            self.costs.append(cost)

            if e % self.meta_params['print_cost_interval'] == 0:
                print('Cost at epoch {0}:  {1}'.format(e, cost))

        if self.meta_params['batch_norm']['use']:
            # Calculate the average bn_Z_var & bn_Z_mean over the whole training set, for future predictions etc
            x, y = self.data_sets['x_training'], self.data_sets['y_training']
            self._forward_prop(x, y, is_training=True)

        self.evaluate('training')

    def predict(self, x):
        """
        Generates predictions for the observations in x. 

        :param x: a numpy array with shape (number_of_input_variables, number_of observations)
        :return: a numpy array (predicted y) with shape (number_of_output_variables, number_of observations)
        """
        self._forward_prop(x=x, y=None, is_training=False)
        return self.model[-1]['Y']

    def evaluate(self, data_set, print_result=True):
        """
        Calculates the average cost for a data set. 
        If print_result==True, then the cost is printed in the console. Otherwise a float is returned. 

        :param data_set: one of the following strings: 'training', 'test', 'validation'
        :return: If print_result==True, then nothing is returned. If print_result==False, then the cost is returned.
        """
        if data_set in ['training', 'test', 'validation']:
            x, y = self.data_sets['x_' + data_set], self.data_sets['y_' + data_set]
            if x.shape[1] > 0:
                self._forward_prop(x, y, is_training=False)
                cost = self.model[-1]['average_cost']
                if print_result:
                    print('Average cost on {0} data set: {1}'.format(data_set, cost))
                else:
                    return cost
            else:
                print('{} set is empty.'.format(data_set))
        else:
            print('data_set must be one of the following: "training", "test", or "validation".')

    def save_net(self, path='NeuralNetModel.pkl', save_model=True, save_params=True, save_data=False):
        """
        Saves data from the NeuralNet object, so that it can be recreated later, using load_net.
        Data is saved as a pickle file.
        :param path: Path to file which should be saved. String. 
        :param save_model: Boolean indicating whether the model should be included. Defaults to True. 
        :param save_params: Boolean indicating whether the meta_params should be included. Defaults to True. 
        :param save_data: Boolean indicating whether the data_sets should be included. Defaults to False. 
        :return: Nothing. 
        """

        net = dict()
        if save_model:
            net['model'] = self.model
        if save_params:
            net['meta_params'] = self.meta_params
        if save_data:
            net['data_sets'] = self.data_sets

        with open(path, 'wb') as nn_file:
            pickle.dump(net, nn_file)

    def load_net(self, path):
        """
        Loads model, parameters and/or data to the NeuralNet object from a pickle file. 
        This pickle file has typically been created using the save_net method. 
        :param path: string. 
        :return: Nothing. 
        """

        with open(path, 'rb') as nn_file:
            net = pickle.load(nn_file)

        if type(net) == dict:
            loaded = list()
            if 'model' in net.keys():
                self.model = net['model']
                loaded.append('model')
            if 'meta_params' in net.keys():
                self.meta_params = net['meta_params']
                loaded.append('meta_params')
            if 'data_sets' in net.keys():
                self.data_sets = net['data_sets']
                loaded.append('data_sets')
            if 'costs' in net.keys():
                self.costs = net['costs']
                loaded.append('costs')
            if 'cost_counter' in net.keys():
                self.cost_counter = net['cost_counter']
                loaded.append('cost_counter')
            print('Loaded from {0}: {1}'.format(path, ', '.join(loaded)))

    def clean_model(self, remove_nonvital_params=False, convert_batch_norm=False):
        """
        If remove_non-vital_params is True, then all items in the model that were created during training are removed, 
        to leave only model parameters etc that are needed for predictions. (Optimization data, caches etc are removed.)
        Note that after this, you may not be able to retrain the model, without reinitializing the parameters. 

        If convert_batch_norm is True, then the model parameters are changed from [W, bn_gamma, bn_beta] to [W, b] in 
        all lin_act layers. model['batch_norm']['use'] is also set to False. 

        :return: Nothing. 
        """

        # Remove unnecessary variables and optimization parameters
        if remove_nonvital_params:
            params_to_keep = ['type', 'W', 'b', 'bn_gamma', 'bn_beta', 'bn_Z_mean', 'bn_Z_var', 'units', 'activation',
                              'cost_function']
            for layer in self.model:
                for key in list(layer.keys()):
                    if key not in params_to_keep:
                        layer.pop(key, None)

        # Convert batch-norm parameters to normal parameters. Replace W, bn_beta & bn_gamma with W & b.
        if convert_batch_norm and self.meta_params['batch_norm']['use']:
            for layer in self.model:
                if layer['type'] == 'lin_act':
                    z_std = np.sqrt(layer['bn_Z_var'] + self.meta_params['batch_norm']['epsilon'])
                    layer['W'] = layer['W'] * layer['bn_gamma'] / z_std
                    layer['b'] = layer['bn_beta'] - layer['bn_gamma'] * layer['bn_Z_mean'] / z_std
                    for k in ['bn_beta', 'bn_gamma', 'bn_Z_mean', 'bn_Z_var']:
                        layer.pop(k, None)

            # Turn off batch norm in meta parameters
            self.meta_params['batch_norm']['use'] = False

    def _get_mini_batches(self):
        # Permute data
        data = self.data_sets
        permutation = list(np.random.permutation(data['x_training'].shape[1]))
        x = data['x_training'][:, permutation].reshape((data['x_training'].shape[0], data['x_training'].shape[1]))
        y = data['y_training'][:, permutation].reshape((data['y_training'].shape[0], data['y_training'].shape[1]))

        # Cut training set into mini-batches
        mini_batch_size = self.meta_params['optim']['mini_batch_size']
        # num_mini_batches = math.ceil(data['x_training'].shape[1] / mini_batch_size)
        num_mini_batches = int((data['x_training'].shape[1] - 1) / mini_batch_size) + 1
        batches = list()
        for k in range(0, int(num_mini_batches)):
            batches.append({
                'x': data['x_training'][:, k * mini_batch_size: (k + 1) * mini_batch_size],
                'y': data['y_training'][:, k * mini_batch_size: (k + 1) * mini_batch_size]
            })
        return batches

    def _forward_prop(self, x, y, is_training=False):
        bn_use = self.meta_params['batch_norm']['use']
        if bn_use:
            bn_epsilon = self.meta_params['batch_norm']['epsilon']
        previous = x
        for layer in self.model:
            layer['previous'] = previous
            if layer['type'] == 'lin_act':
                # Linear transformation
                if bn_use:
                    _Z = np.dot(layer['W'], previous)
                    if is_training:
                        layer['bn_Z_mean'] = _Z.mean(axis=1, keepdims=True)
                        layer['bn_Z_var'] = _Z.var(axis=1, ddof=1, keepdims=True)
                    layer['Z_norm'] = (_Z - layer['bn_Z_mean']) / np.sqrt(layer['bn_Z_var'] + bn_epsilon)
                    Z = layer['bn_gamma'] * layer['Z_norm'] + layer['bn_beta']
                else:
                    Z = np.dot(layer['W'], previous) + layer['b']

                layer['Z'] = Z

                # Activation function
                A = Z
                if layer['activation'] == 'relu':
                    A = np.maximum(Z, 0)
                elif layer['activation'] == 'sigmoid':
                    A = np.divide(1, (1 + np.exp(-Z)))
                # elif layer['activation'] == 'tanh':
                #    A = np.tanh(Z)
                layer['A'] = A
                previous = A
            # elif layer['type'] == 'softmax':
            #     p = np.exp(previous - np.max(previous))
            #     layer['P'] = p / np.sum(p, axis=0, keepdims=True)
            #     previous = layer['P']
            elif layer['type'] == 'cost':
                layer['Y'] = previous
                if y is not None:
                    self._calculate_cost(layer, previous, y)
                    # No need tos save into previous, as there is no next layer
            elif layer['type'] == 'input':
                pervious = x

    def _backward_prop(self, x, y):
        regularization_lambda = self.meta_params['regularization_lambda']
        m = x.shape[1]  # number of observations
        next_grad = self.model[-1]['d_prev']
        for layer in reversed(self.model[:-1]):
            if layer['type'] == 'lin_act':
                # Calculate gradient of activation function input (based on activation gradient d_A)
                # Note: d_A is stored in next_grad
                if layer['activation'] == 'relu':
                    d_Z = next_grad * (np.greater(layer['A'], 0) * 1.0)
                # elif layer['activation'] == 'tanh':
                #     d_Z = next_grad * np.greater(layer['A'], 0) * 1.0   # TODO: tanh
                elif layer['activation'] == 'sigmoid':
                    s = np.divide(1, (1 + np.exp(-layer['A'])))
                    d_Z = next_grad * s * (1 - s)

                # Calculate gradients of parameters in the linear transformation
                # Note: There are two cases: with or without batch norm
                if self.meta_params['batch_norm']['use']:
                    # d_bn_beta & d_bn_gamma
                    layer['d_bn_beta'] = np.sum(d_Z, axis=1, keepdims=True) / m
                    layer['d_bn_gamma'] = np.sum(d_Z * layer['Z_norm'], axis=1, keepdims=True) / m

                    # d_W & d_prev
                    Z_var = layer['bn_Z_var'] + self.meta_params['batch_norm']['epsilon']
                    Z_std = np.sqrt(Z_var)
                    Z_centered = layer['Z_norm'] * Z_std
                    d_Z_ = layer['bn_gamma'] / m / Z_std * (
                        m * d_Z - np.sum(d_Z, axis=1, keepdims=True) - Z_centered * np.sum(d_Z * Z_centered, axis=1,
                                                                                           keepdims=True) / Z_var)
                    layer['d_W'] = (np.dot(d_Z_, layer['previous'].T) + regularization_lambda * layer['W']) / m
                    layer['d_prev'] = np.dot(layer['W'].T, d_Z_)
                else:
                    layer['d_W'] = (np.dot(d_Z, layer['previous'].T) + regularization_lambda * layer['W']) / m
                    layer['d_b'] = np.sum(d_Z, axis=1, keepdims=True) / m
                    layer['d_prev'] = np.dot(layer['W'].T, d_Z)

                # Cache the gradient to pass back to prior layer
                next_grad = layer['d_prev']

            elif layer['type'] == 'softmax':
                # next_grad =  # TODO
                print('ERROR: SOFTMAX not yet supported.')
            elif layer['type'] == 'cost':
                print('ERROR: Cost layer found in wrong place. Only the last layer should be a cost function.')

    def _calculate_cost(self, layer, A_prev, Y):
        # Calculate cost per observation, using the relevant cost function
        if layer['cost_function'] == 'cross-entropy':
            # layer['cost'] = (-1 / Y.shape[1]) * np.sum(
            #     np.multiply(Y, np.log(A_prev)) + np.multiply(1 - Y, np.log(1 - A_prev)))
            eps = 1e-7
            layer['cost'] = -(np.multiply(Y, np.log(A_prev + eps)) + np.multiply((1. - Y), np.log(1. - A_prev + eps)))
            layer['d_prev'] = -(np.divide(Y, A_prev + eps) - np.divide((1. - Y), (1. - A_prev + eps)))

        elif layer['cost_function'] == 'mean_squared_error':
            layer['cost'] = np.square((A_prev - Y))
            layer['d_prev'] = 2 * (A_prev - Y)

        # Reduce to average cost per observation
        layer['average_cost'] = np.mean(layer['cost'])

        # Save sum and number of observations
        count = layer['cost'].shape[-1]
        self.cost_counter['count'] += count
        self.cost_counter['sum'] += layer['average_cost'] * count

    def _update_parameters(self):
        if self.meta_params['optim']['algo'] == 'adam':
            # Get parameters
            self.meta_params['optim']['update_counter'] += 1.0
            update_counter = self.meta_params['optim']['update_counter']
            learning_rate = self.meta_params['optim']['learning_rate']
            beta1 = self.meta_params['optim']['beta1']
            beta2 = self.meta_params['optim']['beta2']
            epsilon = self.meta_params['optim']['epsilon']

            for layer in self.model:
                if layer['type'] == 'lin_act':
                    for param in [p for p in ['W', 'b', 'bn_gamma', 'bn_beta'] if p in layer]:
                        layer['v_' + param] = beta1 * layer['v_' + param] + (1.0 - beta1) * layer['d_' + param]
                        _v = layer['v_' + param] / (
                            1.0 - beta1 ** update_counter)  # 1st moment (corrected for early obs)
                        layer['s_' + param] = beta2 * layer['s_' + param] + (1.0 - beta2) * layer['d_' + param] ** 2.0
                        _s = layer['s_' + param] / (1.0 - beta2 ** update_counter)  # 2nd moment (corrected)
                        layer[param] -= learning_rate * _v / (_s ** 0.5 + epsilon)  # Update

        elif self.meta_params['optim']['algo'] == 'gradient_descent':
            # Get parameters
            learning_rate = self.meta_params['optim']['learning_rate']
            beta = self.meta_params['optim']['beta']

            for layer in self.model:
                if layer['type'] == 'lin_act':
                    for param in [p for p in ['W', 'b', 'bn_gamma', 'bn_beta'] if p in layer]:
                        layer['v_' + param] = beta * layer['v_' + param] + (1 - beta) * layer['d_' + param]
                        layer[param] -= learning_rate * layer['v_' + param]

    def _reset_cost_counter(self):
        self.cost_counter = {'sum': 0.0, 'count': 0.0}

