A method to identify concept drifts in data using a Generative Adversarial Network and a feed forward network. 
For the code to execute, please download the required datasets from https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow
and copy it into a folder called datasets.

For executing the code, the name of the dataset can be passed to the read_dataset function.
Possible names for the datasets are 'airlines', 'chess', 'electric', 'ludata', 'outdoor', 'phishing', 
'poker', 'rialto', 'spam', 'weather', 'interchanging_rbf', 'mixed_drfit', 'moving_rbf', 'moving_squares', 
'rotating_hyperplane', 'sea_big' and 'transient_chessboard'.

Default parameters are
initial_limit = 100
epochs = 200
batch_size = 4
threshold = 0.53
