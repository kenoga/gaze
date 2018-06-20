
import json
import pandas
with open('results.json', 'r') as fr:
    data = json.load(fr)
print(data)
import matplotlib.pyplot as plt

# Set values
x = list(range(1, len(data['train']['loss'])+1))
y0 = data['train']['loss']
y1 = data['train']['accuracy']
y2 = data['test']['loss']
y3 = data['test']['accuracy']
print(x)
print(y0)
# Set background color to white
fig = plt.figure()
fig.patch.set_facecolor('white')

# Plot lines
plt.xlabel('epoch')
plt.plot(x, y0, label='train_loss')
plt.plot(x, y1, label='train_accuracy')
plt.plot(x, y2, label='test_loss')
plt.plot(x, y3, label='test_accuracy')
plt.legend()

# Visualize
plt.show()