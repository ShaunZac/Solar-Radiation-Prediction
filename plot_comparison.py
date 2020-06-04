# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# data obtained from the other files
nn_mae = [16.280746459960938, 20.573495864868164, 21.346220016479492]
svm_mae = [14.888599272664527, 17.814461634294563, 19.22303075715943]
nn_mape = [25.192826921530278, 22.915396391128118, 22.40351912511298]
svm_mape = [23.191380898353525, 21.917128945561726, 22.351748177279607]

index = np.array([i for i in range(1, 4)])
bar_width = 0.2

# plotting MAE
plt.figure()
fig, ax = plt.subplots()
plt.ylim(0, 28)
nn = ax.bar(index, nn_mae, bar_width, label='Neural Network')
svm = ax.bar(index+bar_width, svm_mae, bar_width, label='Suport Vector Machine')
ax.set_xlabel("Prediction Horizon (hours)")
ax.set_ylabel("MAE $(W/m^2)$")
ax.set_title("Mumbai")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["1", '2', '3'])
ax.legend()
plt.show()

# plotting MAPE
plt.figure()
fig, ax = plt.subplots()
plt.ylim(0, 30)
nn = ax.bar(index, nn_mape, bar_width, label='Neural Network')
svm = ax.bar(index+bar_width, svm_mape, bar_width, label='Suport Vector Machine')
ax.set_xlabel("Prediction Horizon (hours)")
ax.set_ylabel("MAPE (%)")
ax.set_title("Mumbai")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["1", '2', '3'])
ax.legend()
plt.show()
