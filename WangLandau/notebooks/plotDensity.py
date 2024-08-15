import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict

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


def get_renormalized_log_g_values(results: dict):
    results_x = []
    results_y = []
    for result in results:
        x_values = np.array(list(result.keys()))
        y_values = np.array(list(result.values()))
        normalized_values = y_values - np.min(y_values)
        results_y.append(normalized_values)
        results_x.append(x_values)
    return results_x, results_y


def get_renormalized_log_g_values_as_dict_list(results):
    normalized_results = []
    for result in results:
        x_values = np.array(list(result.keys()))
        y_values = np.array(list(result.values()))
        normalized_values = y_values - np.min(y_values)
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
            continue  # Skip if there are no common keys -> this must hold for all intervals such that result returns empty set
        
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
    if(len(minimum_deviation_energies)!= 0):
        for i, e_concat in enumerate(minimum_deviation_energies):

            e_concat_index_in_preceeding_interval = np.where(results_x[i] == e_concat)[0]
            e_concat_index_in_following_interval = np.where(results_x[i+1] == e_concat)[0]

            if e_concat_index_in_preceeding_interval.size == 0 or e_concat_index_in_following_interval.size == 0:
                raise ValueError(f"e_concat {e_concat} not found in one of the intervals.")

            e_concat_index_in_preceeding_interval = e_concat_index_in_preceeding_interval[0]
            e_concat_index_in_following_interval = e_concat_index_in_following_interval[0]

            shift_val = (results_y[i][e_concat_index_in_preceeding_interval] - results_y[i+1][e_concat_index_in_following_interval])

            """resclaing for continous concat"""
            results_y[i+1] += shift_val

            # """cutting the overlapping parts"""
            # results_y[i] = results_y[i][:e_concat_index_in_preceeding_interval+1]
            # results_x[i] = results_x[i][:e_concat_index_in_preceeding_interval+1]
            # results_y[i+1] = results_y[i+1][e_concat_index_in_following_interval:]
            # results_x[i+1] = results_x[i+1][e_concat_index_in_following_interval:]
    else:
        for i in range(len(results_x)-1):
            shift_val = (results_y[i][-1] - results_y[i+1][0])
            """resclaing for   concat"""
            results_y[i+1] += shift_val
    return

def extract_theory_results_from_file(file_path, N, M):
    M_string = str(M)
    N_string = str(N)
    array = None
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for i, line in enumerate(lines):
            if f"{N_string}_{M_string}" in line:
                numbers_string = re.search(r'\[(.*?)\]', line).group(1)
                array = [int(num.strip()) for num in numbers_string.split(',')]

    if array is None:
        print(f"No array found for N={N} M={M}.")
    return array

def plot_log_g(file_name, exact_results = []):
    walker_results = read_data_from_file(file_name) 

    """normalize the walker results by min value for log results"""
    walker_results = get_renormalized_log_g_values_as_dict_list(walker_results)
    
    """averages over walker results per intervals"""
    walker_results = average_matching_keys(walker_results)

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
    plt.xlabel('E')
    plt.ylabel('log(g)')

    # arg parsing from result name to constrcut plot name
    args_string = file_name.split("results/periodic/")[1].split(".txt")[0].replace("/", "_")
    plt.savefig(f"logPlot_{args_string}.png")

def plot_log_g_diff_to_theory(file_name, exact_results = []):
    walker_results = read_data_from_file(file_name) 

    """normalize the walker results by min value for log results"""
    walker_results = get_renormalized_log_g_values_as_dict_list(walker_results)
    
    """averages over walker results per intervals"""
    walker_results = average_matching_keys(walker_results)

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

    # Step 1: Flatten the lists
    flattened_x = [x for sublist in results_x for x in sublist]
    flattened_y = [y for sublist in results_y for y in sublist]

    # Step 2: Pair x and y values
    xy_pairs = list(zip(flattened_x, flattened_y))

    # Step 3: Remove duplicates by converting the list of pairs into a set
    unique_xy_pairs = list(set(xy_pairs))

    # Step 4: Separate the unique x and y values back into their respective lists
    unique_x = [x for x, y in unique_xy_pairs]
    unique_y = [y for x, y in unique_xy_pairs]  

    exact_results_x = np.linspace(min(unique_x), max(unique_x), len(exact_results))  
    exact_results_y = [np.log(float(max(x/2,0.1))) for x in exact_results]

    plt.figure(figsize=(14, 7))
    for i in range(len(unique_x)):
        plot_data(unique_x[i], unique_y[i], color='b')
    
    plt.plot(exact_results_x, exact_results_y, color = 'r')
    plt.xlabel('E')
    plt.ylabel('log(g)')

    unique_xy_pairs_ordered = list(OrderedDict.fromkeys(xy_pairs))  # Preserves the order
    unique_x_ordered = [x for x, y in unique_xy_pairs_ordered]
    unique_y_ordered = [y for x, y in unique_xy_pairs_ordered]

    print(unique_x_ordered)
    plt.figure()
    for i, x in enumerate(exact_results_x):
        idx = np.where(unique_x_ordered == x)
        if idx[0].size > 0:
            diff = unique_y_ordered[idx[0].tolist()[0]]-exact_results_y[i]
            plot_data(x, diff, color='b')
    plt.ylabel('diff from theoretical')

    plt.show()

    # arg parsing from result name to constrcut plot name
    # args_string = file_name.split("results/periodic/")[1].split(".txt")[0].replace("/", "_")
    # plt.savefig(f"logPlot_{args_string}.png")


def main():
    file_name =  'results/periodic/prob_0.000000/X_12_Y_12/seed_42/error_class_I/intervals_20_iterations_10000_overlap_0.250000_walkers_8_seed_run_42_alpha_0.800000_beta_0.0000000100.txt'
    exact_results = extract_theory_results_from_file("./notebooks/theory_results.txt", 12, 12)
    plot_log_g_diff_to_theory(file_name=file_name, exact_results=exact_results)

if __name__ == '__main__':
    main()