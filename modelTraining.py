from trainNetwork import *
import matplotlib.pyplot as plt


MODEL_DIR = "./models"


# Train the built network
trainNet = trainNetwork()
trainNet.training(featuresVal, resVal, epochs=1)

# Plot the Training Loss
plt.figure()
plt.plot(trainNet.losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()
plt.show()
# Plot the Test Loss
plt.figure()
plt.plot(trainNet.losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()
plt.show()
