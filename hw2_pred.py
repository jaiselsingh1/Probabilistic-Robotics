#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# index 0 means open
# index 1 means closed
# rows are measurements, columns are priors
measurement_model = np.array(
    [[0.9, 0.5],
     [0.1, 0.5]]
)

bel = np.array(
    [[0.5],
     [0.5]]
)

# Second belief state for no prediction case
bel_no_pred = np.array(
    [[0.5],
     [0.5]]
)

measurements = [0, 0, 0, 0, 0, 1, 0, 0]

# Track belief progression
belief_history_with_prediction = []
belief_history_no_prediction = []

def prediction_model(prev_belief):
    if prev_belief[0, 0] > 0.9:
        M = np.array([[1, 0.6], [0, 0.4]])
        predicted_belief = M @ prev_belief
    else:
        M = np.eye(2)
        predicted_belief = M @ prev_belief
    return predicted_belief

for measurement in measurements:
    # With prediction steps
    bel = prediction_model(bel)
    unnormalized_posterior = (
        measurement_model * \
        np.repeat(bel, 2, axis=1).transpose())[measurement]
    posterior = unnormalized_posterior / sum(unnormalized_posterior)
    
    belief_open = round(posterior[0], 3)
    belief_history_with_prediction.append(belief_open)
    
    bel = np.array([posterior]).transpose()
    
    # Without prediction steps (measurement only)
    unnormalized_posterior_no_pred = (
        measurement_model * \
        np.repeat(bel_no_pred, 2, axis=1).transpose())[measurement]
    posterior_no_pred = unnormalized_posterior_no_pred / sum(unnormalized_posterior_no_pred)
    
    belief_open_no_pred = round(posterior_no_pred[0], 3)
    belief_history_no_prediction.append(belief_open_no_pred)
    
    bel_no_pred = np.array([posterior_no_pred]).transpose()


plt.figure(figsize=(12, 6))

# Plot with prediction
plt.plot(range(1, len(belief_history_with_prediction) + 1), belief_history_with_prediction, 
         'b-o', linewidth=2, markersize=8, label='With Prediction Steps')

# Plot without prediction  
plt.plot(range(1, len(belief_history_no_prediction) + 1), belief_history_no_prediction, 
         'r-s', linewidth=2, markersize=8, label='Measurement Only (No Prediction)')

plt.xlabel('Measurement Step')
plt.ylabel('Belief (Door Open)')
plt.title('Belief Progression: With vs Without Prediction Steps')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

# Add measurement labels
measurement_labels = ['open', 'open', 'open', 'open', 'open', 'closed', 'open', 'open']
plt.xticks(range(1, len(belief_history_with_prediction) + 1), measurement_labels, rotation=45)
plt.yticks(np.arange(0, 1.01, 0.05))  # Every 0.05 from 0 to 1
plt.tight_layout()
plt.show()