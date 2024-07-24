import matplotlib.pyplot as plt

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

def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Data from File')
    plt.grid(True)
    plt.savefig("densityDistribution.png")

def main():
    filename = 'WangLandau/log_density_afterRun.txt'  # Replace with your file path
    x, y = read_data_from_file(filename)
    plot_data(x, y)

if __name__ == '__main__':
    main()