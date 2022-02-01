import pandas as pd
import matplotlib.pyplot as plt

path = "09_pruning/"
data = pd.read_csv(path + "results_inference_time.csv", delimiter=",")
print(data.head())
data["Inference time (s)"] = (data["1"] + data["2"]) / 2
data = data[["Pruningrate", "Inference time (s)"]]
print(data.head())
plt.style.use("seaborn")
# data.plot(x="Pruningrate", y="Inference time (s)", grid=True, title="Pruned models vs Inference time on Lenovo Thinkpad",
# figsize=(10, 5), xlabel="Models with Percentages of weights pruned", ylabel="Inference time for 1000 samples (s)")
fig = plt.figure()
fig = plt.title("Pruned models vs Inference time on Lenovo Thinkpad", fontsize=16)
plt.subplot(111)
plt.xlabel("Models with Percentages of weights pruned")
plt.ylabel("Inference time for 1000 samples (s)")
plt.plot(data["Pruningrate"], data["Inference time (s)"], "g")
plt.plot(data["Pruningrate"], data["Inference time (s)"], "og")
plt.xticks(ticks=data["Pruningrate"])
plt.grid(True)
# plt.legend(loc='best')
plt.show()
