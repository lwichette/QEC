import re
import json
from collections import defaultdict
import mpmath as mp
import scipy

mp.mp.dps = 30  # higher precision


def log_sum_exp(to_sum):
    maxval = max(to_sum)
    exp_sum = 0
    for value in to_sum:
        exp_sum += mp.exp(value - maxval)
    res = maxval + mp.log(exp_sum)
    return res


# Free energy given histogram and temperature, arbitrary precision governed by mp
def free_energy(E_list, log_g_list, T):
    to_sum = []
    for i, log_g in enumerate(log_g_list):
        to_sum.append(log_g - E_list[i] / T)
    maxval = max(to_sum)
    exp_sum = 0
    for value in to_sum:
        exp_sum += mp.exp(value - maxval)
    res = maxval + mp.log(exp_sum)
    return -T * res


# Run over batch of results, structured by seed, then by class
def get_free_energies(rescaled_results, temperatures):
    free_energies = []
    for seed_results in rescaled_results:
        free_energy_classes = []
        for error_result in seed_results:
            f_values = []
            for T in temperatures:
                f_values.append(free_energy(error_result[0], error_result[1], T) / (-T))
            free_energy_classes.append(f_values)
        free_energies.append(free_energy_classes)
    return free_energies


def read_results_file(path):

    with open(path, "r") as file:
        content = file.read()

    content = content.strip().rstrip(",")

    corrected_json = f"[{content}]"

    try:
        data = json.loads(corrected_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")

    return data


def process_data(data, batch_results, p, X, Y, error):
    for entry in data:
        histogram_seed = entry["histogram_seed"]
        run_seed = entry["run_seed"]
        results = entry["results"]

        E_list = []
        log_g_list = []

        # Process the results
        for key, value in results.items():
            E_list.append(int(key))
            log_g_list.append(float(value))

        batch_results.append(
            {
                "prob": p,
                "X": X,
                "Y": Y,
                "error": error,
                "histogram_seed": histogram_seed,
                "run_seed": run_seed,
                "E": E_list,
                "log_g": log_g_list,
            }
        )


def filter_and_normalize_rbim(batch_results):
    # grouped dictionary with keys prob size and hist seed
    grouped_results = defaultdict(list)
    for result in batch_results:
        key = (result["prob"], result["X"], result["Y"], result["histogram_seed"])
        grouped_results[key].append(result)

    filtered_results = defaultdict(list)
    for key, results in grouped_results.items():
        newkey = (key[0], key[1], key[1])
        errors = set(result["error"] for result in results)
        if errors == {"I", "X", "Y", "Z"}:
            # To be removed once normalization is properly handled in c
            for result in results:
                log_g_list = result["log_g"]
                offset = log_sum_exp(log_g_list)
                rescaled_log_g_list = [
                    res + mp.log(2) * key[1] * key[2] - offset for res in log_g_list
                ]
                result["log_g"] = rescaled_log_g_list
            filtered_results[newkey].append(
                [[result["E"], result["log_g"]] for result in results]
            )
        else:
            print(
                f"has issue with an error class prob: {key[0]} X: {key[1]} Y: {key[2]} interaction seed: {key[2]} available errors: {errors}"
            )

    return filtered_results


def calc_free_energies_batch(filtered_results, probability, X, Y):
    T_Nish = 1 / (mp.log((1 - probability) / probability) / 2)
    temperatures = [1e-20, T_Nish, 1e20]
    batch_res = filtered_results[(probability, X, Y)]
    free_energies = get_free_energies(batch_res, temperatures)
    print(
        "Number of seeds at p", probability, ", X", X, ", Y", Y, ":", len(free_energies)
    )
    return free_energies


def calculate_curve(free_energies, temp):
    """Calculate the curve value for a given temperature."""
    return 1 - mp.fsum(
        [
            f_class[0][temp] < f_class[1][temp]
            or f_class[0][temp] < f_class[2][temp]
            or f_class[0][temp] < f_class[3][temp]
            for f_class in free_energies
        ]
    ) / len(free_energies)


def calculate_bounds(successes, failures, curve):
    """Calculate the confidence interval bounds."""
    lower_bound = curve - scipy.stats.beta.ppf(0.025, 0.5 + successes, 0.5 + failures)
    upper_bound = scipy.stats.beta.ppf(0.025, 0.5 + successes, 0.5 + failures) - curve
    return lower_bound, upper_bound


def get_optimal_curves(free_energies):
    """Calculate optimal and T0 curves and their confidence intervals."""

    # Calculate curves for two different temperatures
    optimal_curve = calculate_curve(free_energies, temp=1)  # Nishimori temp
    T0_curve = calculate_curve(free_energies, temp=0)  # Low temp

    # Number of successes and failures for both curves
    total_len = len(free_energies)
    number_success_optimal = round(total_len * optimal_curve)
    number_failure_optimal = total_len - number_success_optimal

    number_success_T0 = round(total_len * T0_curve)
    number_failure_T0 = total_len - number_success_T0

    # Calculate confidence interval bounds
    lower_bounds_T0, upper_bounds_T0 = calculate_bounds(
        number_success_T0, number_failure_T0, T0_curve
    )
    lower_bounds_optimal, upper_bounds_optimal = calculate_bounds(
        number_success_optimal, number_failure_optimal, optimal_curve
    )

    return {
        "optimal_curve": optimal_curve,
        "T0_curve": T0_curve,
        "lower_bounds_T0": lower_bounds_T0,
        "upper_bounds_T0": upper_bounds_T0,
        "lower_bounds_optimal": lower_bounds_optimal,
        "upper_bounds_optimal": upper_bounds_optimal,
    }
