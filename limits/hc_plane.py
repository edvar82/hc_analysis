import matplotlib.pyplot as plt

def load_quota(quota_file_name):
    # Read the data from the text file
    data = []
    with open(quota_file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith('>>>'):
                try:
                    values = line.split()
                    data.append((float(values[0]), float(values[1])))
                except ValueError:
                    continue

    # Extract HT and CJT values
    ht = [d[0] for d in data]
    cjt = [d[1] for d in data]
    return ht, cjt

ht, cjt =  load_quota('trozos-N6.q1')
ht1, cjt1 =  load_quota('trozos-N24.q1')
ht2, cjt2 = load_quota('trozos-N120.q1')
ht3, cjt3 = load_quota('trozos-N720.q1')

ht4, cjt4 = load_quota('continua-N6.q1')
ht5, cjt5 = load_quota('continua-N24.q1')
ht6, cjt6 = load_quota('continua-N120.q1')
ht7, cjt7 = load_quota('continua-N720.q1')

# Plot the HT x CJT graph
plt.plot(ht, cjt)
plt.plot(ht1, cjt1)

plt.plot(ht2, cjt2)
plt.plot(ht3, cjt3)
# plt.plot(ht4, cjt4)
# plt.plot(ht5, cjt5)
# plt.plot(ht6, cjt6)
plt.plot(ht7, cjt7)

plt.xlabel('CJT (Complexity of Jensen-Tsallis)')
plt.ylabel('HT (Hurst exponent)')
plt.title('HT x CJT Graph')
plt.grid(True)
plt.show()