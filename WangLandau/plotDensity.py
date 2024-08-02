import matplotlib.pyplot as plt
import numpy as np


def read_data_from_file(filename):

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


def get_renormalized_log_g_values_as_dict_list(results):
    normalized_results = []
    for result in results:
        x_values = np.array(list(result.keys()))
        y_values = np.array(list(result.values()))
        normalized_values = y_values / np.sum(y_values)
        normalized_results.append(dict(zip(x_values, normalized_values)))
    return normalized_results


def average_matching_keys(dict_list):
    from collections import defaultdict
    
    # Group dictionaries by their keys
    grouped_dicts = defaultdict(list)
    for d in dict_list:
        key_tuple = tuple(sorted(d.keys()))
        grouped_dicts[key_tuple].append(d)
    
    # Calculate average for each group of dictionaries
    result_list = []
    for key_tuple, dicts in grouped_dicts.items():
        avg_dict = {}
        for key in key_tuple:
            avg_dict[key] = sum(d[key] for d in dicts) / len(dicts)
        result_list.append(avg_dict)
    
    return result_list

def get_derivative_wrt_e(walker_results, precision=5):
    derivatives = []
    for result in walker_results:
        derivative = {}
        keys = sorted(result.keys())
        for i in range(1, len(keys)):
            x1, x2 = keys[i-1], keys[i]
            y1, y2 = result[x1], result[x2]
            # Compute the derivative
            deriv = (y2 - y1) / (x2 - x1)
            # store derivative for left bound of tupel
            derivative[x1] = round(deriv, precision)
        derivatives.append(derivative)
    return derivatives

def find_lowest_inverse_temp_deviation(dictionaries):
    # Initialize the result list
    result = []

    # Iterate through neighboring pairs of dictionaries
    for i in range(len(dictionaries) - 1):
        dict1 = dictionaries[i]
        dict2 = dictionaries[i + 1]

        # Find common keys
        common_keys = set(dict1.keys()) & set(dict2.keys())
        
        if not common_keys:
            continue  # Skip if there are no common keys
        
        # Initialize variables to find the minimum deviation
        min_deviation = float('inf')
        best_key = None
        
        # Iterate through common keys to find the one with the smallest deviation
        for key in common_keys:
            value1 = dict1[key]
            value2 = dict2[key]
            deviation = abs(value1 - value2)
            
            if deviation < min_deviation:
                min_deviation = deviation
                best_key = key
        
        # Append the best key for this pair to the result list
        result.append(best_key)
    return result

def rescale_results_for_concatenation(results_x, results_y, minimum_deviation_energies):
    for i, e_concat in enumerate(minimum_deviation_energies):
        e_concat_index_in_preceeding_interval = np.where(results_x[i] == e_concat)
        e_concat_index_in_following_interval = np.where(results_x[i+1] == e_concat)
        
        """resclaing for continous concat"""
        results_y[i+1] *= results_y[i][e_concat_index_in_preceeding_interval]/results_y[i+1][e_concat_index_in_following_interval]

        """cutting the overlapping parts"""
        results_y[i] = results_y[i][:e_concat_index_in_preceeding_interval[0][0]+1]
        results_x[i] = results_x[i][:e_concat_index_in_preceeding_interval[0][0]+1]
        results_y[i+1] = results_y[i+1][e_concat_index_in_following_interval[0][0]:]
        results_x[i+1] = results_x[i+1][e_concat_index_in_following_interval[0][0]:]
    return

def main():
    filename =  'WangLandau/results/prob_0.000000/X_10_Y_10/seed_42/log_density_10_10_p0_i12_a07_b1e-6.txt' # 'WangLandau/results/prob_0.000000/X_10_Y_10/seed_42/log_density_10_10_p0_i8_a07_b1e-6.txt' #'WangLandau/results/prob_0.000000/X_10_Y_10/seed_42/log_density_10_10_p0_i4.txt' #
    walker_results = read_data_from_file(filename) 

    """normalize the walker results to sum to 1 to compute averages afterwards"""
    walker_results = get_renormalized_log_g_values_as_dict_list(walker_results)
    
    """averages over walker results per intervals"""
    average_over_intervals_results = average_matching_keys(walker_results)
    walker_results = average_over_intervals_results

    results_x = []
    results_y = []
    for result in walker_results:
        results_y.append(np.array(list(result.values())))
        results_x.append(np.array(list(result.keys())))

    """get inverse temp by log(g) derivative"""
    derivatives_wrt_e = get_derivative_wrt_e(walker_results)

    """get energy per interval pair with lowest deviation of inverse temp"""
    minimum_deviation_energies = find_lowest_inverse_temp_deviation(derivatives_wrt_e)

    """rescaling of log(g) values at concatenation points"""
    rescale_results_for_concatenation(results_x, results_y, minimum_deviation_energies)

    plt.figure(figsize=(14, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(results_x)):
        plot_data(results_x[i], results_y[i], color=colors[i % len(colors)])
    plt.show()


if __name__ == '__main__':
    main()