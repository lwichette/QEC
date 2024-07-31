import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(filename):
    x = []
    y = []

    with open(filename, 'r') as file:
        for line in file:
            # Split the line into X and Y values
            parts = line.split()
            if len(parts) == 2:
                if float(parts[1]) != 0:
                    x_value = float(parts[0])
                    y_value = float(parts[1])
                    x.append(x_value)
                    y.append(y_value)

    return x, y


def read_data_from_file_new(filename):

    walker_results = []

    with open(filename, 'r') as file:
        for line in file:
            data_dict = {}
            data_pairs = line.split(' ,')

            for pair in data_pairs:
                if pair != "\n":
                    x, y = pair.split(' : ')
                    x = int(x.strip())
                    y = float(y.strip())
                    if y != 0:
                        data_dict[x] = y
            walker_results.append(data_dict)

    return walker_results

def plot_data(x, y, color):
    plt.plot(x, y, marker='o', linestyle='-', color=color)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Data from File')
    plt.grid(True)

def get_renormalized_g_values(results: dict):
    exponential_results_x = []
    exponential_results_y = []
    for result in results:
        x_values = np.array(list(result.keys()))
        y_values = np.array(list(result.values()))
        max_y = np.max(y_values)
        adjusted_y_values = y_values - max_y
        exp_values = np.exp(adjusted_y_values)
        normalized_values = exp_values / np.sum(exp_values)
        exponential_results_y.append(normalized_values)
        exponential_results_x.append(x_values)
    return exponential_results_x, exponential_results_y

def get_renormalized_log_g_values(results: dict):
    results_x = []
    results_y = []
    for result in results:
        x_values = np.array(list(result.keys()))
        y_values = np.array(list(result.values()))
        normalized_values = y_values / np.sum(y_values)
        results_y.append(normalized_values)
        results_x.append(x_values)
    return results_x, results_y

def main():
    filename = 'WangLandau/log_density_afterRun_12_12_p0.txt'  # Replace with your file path
    walker_results = read_data_from_file_new(filename)
    results_x, results_y = get_renormalized_log_g_values(walker_results)

    plt.figure(figsize=(14, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(results_x)):
        plot_data(results_x[i], results_y[i], color=colors[i % len(colors)])
    plt.show()


if __name__ == '__main__':
    main()