from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch import nn, optim
from datasets import read_dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import numpy as np
import torch


def generate_noise(shape):
    """
    Function to generate noise of a required shape.
    :param shape: Tuple that specifies the required shape of noise returned
    :return:
    """
    return np.random.random(size=shape)


def collate(batch):
    """
    Function for collating the batch to be used by the data loader. This function does not handle labels
    :param batch:
    :return:
    """
    # Stack each tensor variable
    x = torch.stack([torch.tensor(x) for x in batch])

    # Return features and labels
    return x, None


def collate_train(batch):
    """
    Function for collating the batch to be used by the data loader. This function does handle labels
    :param batch:
    :return:
    """
    # Stack each tensor variable
    x = torch.stack([torch.tensor(x[:-1]) for x in batch])
    y = torch.stack([torch.tensor([x[-1]]) for x in batch])
    # Return features and labels
    return x, y


def standardize(matrix):
    """
    Function to do a columnwise standardization of an input matrix
    :param matrix: matrix to be standardized
    :return: standardized matrix, mean and standard deviation
    """
    m = matrix.mean(axis=0)
    s = matrix.std(axis=0)
    # If the standard deviation is 0, set those values as 1
    s = [x + int(x == 0) for x in s]
    standardized_matrix = (matrix - m) / s
    return standardized_matrix, m, s


class Generator(Module):
    def __init__(self, inp, out):
        super(Generator, self).__init__()
        self.net = Sequential(Linear(inp, 128), nn.ReLU(),
                              Linear(128, 128), nn.Sigmoid(),
                              Linear(128, out))

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(Module):
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()
        self.net = Sequential(Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

        # self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x


class Network(Module):
    def __init__(self, inp, out):
        super(Network, self).__init__()
        self.net = Sequential(BatchNorm1d(num_features=inp),
                              Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x


def get_discriminator(initial_features, device, epochs=100, steps_generator=100, steps_discriminator=100,
                      seed=0, noisy_input_size=8, batch_size=8, lr=0.0001, momentum=0.9):

    # Set the seed for torch and numpy
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)

    # Create the loss functions for the discriminator and generator
    loss_discriminator = nn.BCELoss()
    loss_generator = nn.BCELoss()

    # Create the generator and discriminator objects and set them to double.
    # The input size to the generator is the noisy input size and the generated vector is the size of a feature vector
    generator = Generator(inp=noisy_input_size, out=initial_features.shape[1])
    generator.double()
    discriminator = Discriminator(inp=initial_features.shape[1], out=1)
    discriminator.double()

    loss_array_generator = []
    loss_array_discriminator = []

    # Create the optimizers for the generator and discriminator
    optimizer_generator = optim.SGD(generator.parameters(), lr=lr, momentum=momentum)
    optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=lr, momentum=momentum)

    # Create the data loader for the features which is the real data
    real_data = DataLoader(initial_features, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Set the generator and discriminator to the GPU/CPU depending on the parameter
    discriminator.to(device)
    generator.to(device)

    # Variables for computing loss of generator and discriminator
    ones = Variable(torch.ones(batch_size, 1)).double().to(device)  # Indicates real data
    zeros = Variable(torch.zeros(batch_size, 1)).double().to(device)  # Indicates generated data

    for epoch in range(epochs):

        total_loss_generator = 0
        total_loss_discriminator = 0

        for step_d in range(steps_discriminator):
            discriminator.zero_grad()
            x = None
            # Train discriminator on actual data
            for real_input, _ in real_data:
                x = real_input
                break

            # Get the loss when the real data is compared to ones
            x = x.to(device)
            output_discriminator = discriminator(x)
            real_loss_discriminator = loss_discriminator(output_discriminator,
                                                         ones)

            # Train discriminator on drifted/noise data
            generator_input = generate_noise(shape=[batch_size, noisy_input_size])

            # Get the output from the generator for the generated data compared to zeroes
            generated_output = generator(torch.Tensor(generator_input).double().to(device))

            generated_output = generated_output.to(device)
            generated_output_discriminator = discriminator(generated_output)
            generated_loss_discriminator = loss_discriminator(generated_output_discriminator, zeros)

            # Add the loss and compute back prop
            total_iter_loss = generated_loss_discriminator + real_loss_discriminator
            total_iter_loss.backward()

            total_loss_discriminator += total_iter_loss

            # Update parameters
            optimizer_discriminator.step()

        for step_g in range(steps_generator):
            generator.zero_grad()

            # Generating data for input to generator
            generated_input = generate_noise(shape=[batch_size, noisy_input_size])
            generated_input = torch.Tensor(generated_input).double().to(device)
            generated_output = generator(generated_input)

            # Compute loss based on whether discriminator can discriminate real data from generated data
            generated_training_discriminator_output = discriminator(generated_output)
            generated_training_discriminator_output = generated_training_discriminator_output.to(device)

            # Compute loss based on ideal target values which are ones
            loss_generated = loss_generator(generated_training_discriminator_output,
                                            ones)

            # Back prop and parameter update
            loss_generated.backward()
            total_loss_generator += loss_generated
            optimizer_generator.step()

        epoch_loss_generator = total_loss_generator.cpu().detach().numpy()/steps_generator
        epoch_loss_discriminator = total_loss_discriminator.cpu().detach().numpy()/steps_discriminator

        loss_array_generator.append(epoch_loss_generator)
        loss_array_discriminator.append(epoch_loss_discriminator)

    return discriminator, generator

        
def train_network(old_features, new_features, batch_size=4, lr=0.0001, momentum=0.9, epochs=100, device="cpu", seed=0):
    # Set the seed for torch and numpy
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)

    # Create the labels for the new and old features
    ones = np.ones(shape=(len(new_features), 1))
    zeros = np.zeros(shape=(len(old_features), 1))
    x_old = np.hstack((old_features, zeros))
    x_new = np.hstack((new_features, ones))
    training_set = np.vstack((x_old, x_new))

    # Initialize the network, and send it to device
    network = Network(inp=old_features.shape[1], out=1)
    network = network.double()
    network = network.to(device)

    dl = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    loss_ = nn.BCELoss()
    for idx in range(epochs):
        for batch_x, batch_y in dl:
            network.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = network(batch_x)
            curr_loss = loss_(out, batch_y)
            curr_loss.backward()
            optimizer.step()

    return network

    
def fit_and_predict(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict([features[0]])
    clf.reset()
    clf.partial_fit([features[0]], [labels[0]], classes=classes)
    for idx in range(1, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf
    
    
def predict_and_partial_fit(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    for idx in range(0, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf
    
    
def find_drifts(features, labels, standardized_features, batch_size, device, epochs, seed,
                initial_limit, threshold=0.53):
    from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
    drift_count = []
    offset = initial_limit
    y_pred = []
    y_true = []

    limit = int(initial_limit/2)

    clf = HoeffdingTreeClassifier()

    # Training window that trains the classifier once a drift occurs
    training_window = []

    x = features[:offset, :]
    y = labels[:offset]

    classes = np.unique(labels)

    predicted, clf = fit_and_predict(clf=clf, features=x, labels=y, classes=classes)
    y_pred = y_pred + predicted.tolist()
    y_true = y_true + y

    generated_input = generate_noise(shape=[limit, 8])
    generated_input = torch.Tensor(generated_input).double().to(device)
    # Train the initial discriminator value
    initial_features = standardized_features[:initial_limit, :]
    discriminator, generator = get_discriminator(initial_features=initial_features, epochs=epochs, seed=seed,
                                                 device=device)

    outliers = generator(generated_input)
    outliers = outliers.cpu().detach().numpy()
    progress_bar = tqdm(total=len(features))

    progress_bar.update(initial_limit)
    threshold = threshold

    while offset + initial_limit < len(features):

        first_batch = torch.from_numpy(standardized_features[offset:offset + batch_size, :]).to(device)

        output = discriminator(first_batch)

        # Check if mean predictions are below threshold
        if np.mean(output.cpu().detach().numpy()) > threshold:
            # Drift is not detected.
            # Predict and fit on the clf
            curr_features = features[offset:offset + batch_size, :]
            curr_labels = labels[offset:offset + batch_size]

            predicted, clf = predict_and_partial_fit(clf=clf, features=curr_features, labels=curr_labels,
                                                     classes=classes)
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + curr_labels

            # Continue to next batch
            offset += batch_size
            progress_bar.update(batch_size)
            continue

        # Drift detected, so retrain classifier
        training_idx_start = offset
        training_idx_end = offset + initial_limit
        # retrain the classifier because of partial drift.

        predicted, clf = fit_and_predict(clf=clf, features=features[training_idx_start:training_idx_end, :],
                                         labels=labels[training_idx_start:training_idx_end],
                                         classes=classes)
        predicted = predicted.tolist()
        y_pred = y_pred + predicted
        y_true = y_true + labels[training_idx_start:training_idx_end]

        # Standardize based on the new data window
        training_idx_end = training_idx_start + initial_limit
        standardized_features_window = features[training_idx_start: training_idx_end, :]
        standardized_features_window, mean, std = standardize(standardized_features_window)
        standardized_features = (features - mean) / std

        # Seed to be changed every time the discriminator is retrained
        seed = np.random.randint(65536)

        # Create the old features based on the old context window.
        old_features = standardized_features[offset - initial_limit:offset, :]

        # retrain the network
        discriminator = train_network(old_features=old_features,
                                      new_features=standardized_features_window, batch_size=16,
                                      device=device, epochs=epochs, seed=seed)

        drift_count.append(offset)

        offset += initial_limit
        progress_bar.update(initial_limit)

    # Test on the remaining features
    features_window = features[offset:, :]
    labels_window = labels[offset:]
    y_hat, clf = predict_and_partial_fit(clf, features=features_window, labels=labels_window, classes=classes)
    y_pred = y_pred + y_hat.tolist()
    y_true = y_true + labels_window

    # Update and close the progress bar
    progress_bar.update(len(features) - offset)
    progress_bar.close()

    return y_pred, y_true, drift_count


def main():
    # Set the number of training instances
    initial_limit = 100

    # Set the number of epochs the GAN should be trained
    epochs = 200

    # Set the batch_size
    batch_size = 4

    # Set the threshold
    threshold = 0.53

    # Read the dataset
    """
    Any one of the following strings can be chosen and it reads the corresponding dataset.
    The dataset should be downloaded from https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow and
    placed in a folder datasets.
    'airlines', 'chess', 'electric', 'ludata', 'outdoor', 'phishing', 'poker', 'rialto', 'spam', 'weather',
     'interchanging_rbf', 'mixed_drfit', 'moving_rbf', 'moving_squares', 'rotating_hyperplane', 'sea_big',
     'transient_chessboard'
    """
    features, labels = read_dataset('moving_squares')
    features = np.array(features)

    # Get the device the experiment will run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the initial training set and standardize it
    initial_features = features[:initial_limit, :]
    initial_features, mean, std = standardize(initial_features)

    # Standardize the whole dataset
    standardized_features = features / std

    # Set a random seed for the experiment
    seed = np.random.randint(65536)
    print('The seed for the current execution is %d ' % seed)

    t1 = time()
    y_pred, returned_labels, drifts = find_drifts(features=features, standardized_features=standardized_features,
                                                  labels=labels, epochs=epochs, initial_limit=initial_limit,
                                                  device=device, seed=seed, batch_size=batch_size,
                                                  threshold=threshold)
    t2 = time()

    # Get the auc_value
    from sklearn.metrics import accuracy_score
    auc_value = accuracy_score(y_true=returned_labels, y_pred=y_pred)
    print(' Accuracy value is %f' % auc_value)

    exec_time = t2 - t1
    print('Execution time is %d seconds' % exec_time)


if __name__ == '__main__':
    main()
