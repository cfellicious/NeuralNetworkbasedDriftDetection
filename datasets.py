import csv


def read_dataset(name='electric'):
    path_dict = {'airlines': './datasets/real-world/airlines.csv',
                 'chess': './datasets/real-world/chessweka.csv',
                 'electric': './datasets/real-world/elec.csv',
                 'ludata': './datasets/real-world/LUdata.csv',
                 'outdoor': './datasets/real-world/outdoorStream.csv',
                 'phishing': './datasets/real-world/phishing.csv',
                 'poker': './datasets/real-world/poker.csv',
                 'rialto': './datasets/real-world/rialto.csv',
                 'spam': './datasets/real-world/spam.csv',
                 'weather': './datasets/real-world/weather.csv',
                 'interchanging_rbf': './datasets/artificial/interchangingRBF.csv',
                 'mixed_drfit': './datasets/artificial/mixeddrift.csv',
                 'moving_rbf': './datasets/artificial/movingRBF.csv',
                 'moving_squares': './datasets/artificial/interchangingRBF.csv',
                 'rotating_hyperplane': './datasets/artificial/rotatingHyperplane.csv',
                 'sea_big': './datasets/artificial/sea_big.csv',
                 'transient_chessboard': './datasets/artificial/transientChessboard.csv'
                 }

    path = path_dict.get(name.lower())
    array = []
    labels = []

    with open(path, 'r') as f:
        lines = csv.reader(f)
        # Skip headers
        next(lines)
        for line in lines:
            temp = [float(x) for x in line]
            labels.append(int(temp[-1]))
            temp = temp[:-1]
            array.append(temp)

    return array, labels
