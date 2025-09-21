import numpy as np
import matplotlib.pyplot as plt
#measurement model
M = np.array([
    [0.9, 0.5],  
    [0.1, 0.5]   
])

#action model
M_Push = np.array([
    [1.0, 0.6], 
    [0.0, 0.4]  
])

#initial conditions
bel = np.array([[0.5], [0.5]])

measurements = [0, 0, 0, 0, 0, 1, 0, 0]

history_open = [bel[0,0]]
history_closed = [bel[1,0]]

#running filter with action
for t, measurement in enumerate(measurements):
    bel_bar = M_Push @ bel #apply action model
    # measurement update
    likelihood = M[measurement].reshape(2,1)
    unnormalized_posterior = likelihood * bel_bar
    bel = unnormalized_posterior / unnormalized_posterior.sum()
    history_open.append(bel[0,0])
    history_closed.append(bel[1,0])
    print(f"step {t}: bel(open)={bel[0,0]:.3f}, bel(closed)={bel[1,0]:.3f}")

bel = [0.5, 0.5]  # reset belief
history_open_noaction = [bel[0]]
history_closed_noaction = [bel[1]]

for t,m in enumerate(measurements):
    # measurement update only
    likelihood = M[m]
    unnormalized_posterior = likelihood * bel
    bel = unnormalized_posterior / np.sum(unnormalized_posterior)
    history_open_noaction.append(bel[0])
    history_closed_noaction.append(bel[1])
    print(f"step {t} (no action): bel(open)={bel[0]:.3f}, bel(closed)={bel[1]:.3f}")

steps = range(len(history_open))
plt.figure(figsize=(8,5))
plt.plot(steps, history_open, marker='o', label="bel(open) with push action")
plt.plot(steps,history_open_noaction, marker='x', label="bel(open) no action")
plt.xlabel("Time step")
plt.ylabel("Belief")
plt.title("Belief progression with and without action")
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.show()




# #!/usr/bin/env python
# import numpy as np
# import matplotlib.pyplot as plt

# # index 0 means open
# # index 1 means closed
# # rows are measurements, columns are priors
# measurement_model = np.array(
#     [[0.9, 0.5],
#      [0.1, 0.5]]
# )

# bel = np.array(
#     [[0.5],
#      [0.5]]
# )

# # Second belief state for no prediction case
# bel_no_pred = np.array(
#     [[0.5],
#      [0.5]]
# )

# measurements = [0, 0, 0, 0, 0, 1, 0, 0]

# # Track belief progression
# belief_history_with_prediction = []
# belief_history_no_prediction = []

# def prediction_model(prev_belief):
#     if prev_belief[0, 0] > 0.9:
#         M = np.array([[1, 0.6], [0, 0.4]])
#         predicted_belief = M @ prev_belief
#     else:
#         M = np.eye(2)
#         predicted_belief = M @ prev_belief
#     return predicted_belief

# for measurement in measurements:
#     # With prediction steps
#     bel = prediction_model(bel)
#     unnormalized_posterior = (
#         measurement_model * \
#         np.repeat(bel, 2, axis=1).transpose())[measurement]
#     posterior = unnormalized_posterior / sum(unnormalized_posterior)
    
#     belief_open = round(posterior[0], 3)
#     belief_history_with_prediction.append(belief_open)
    
#     bel = np.array([posterior]).transpose()
    
#     # Without prediction steps (measurement only)
#     unnormalized_posterior_no_pred = (
#         measurement_model * \
#         np.repeat(bel_no_pred, 2, axis=1).transpose())[measurement]
#     posterior_no_pred = unnormalized_posterior_no_pred / sum(unnormalized_posterior_no_pred)
    
#     belief_open_no_pred = round(posterior_no_pred[0], 3)
#     belief_history_no_prediction.append(belief_open_no_pred)
    
#     bel_no_pred = np.array([posterior_no_pred]).transpose()


# plt.figure(figsize=(12, 6))

# # Plot with prediction
# plt.plot(range(1, len(belief_history_with_prediction) + 1), belief_history_with_prediction, 
#          'b-o', linewidth=2, markersize=8, label='With Prediction Steps')

# # Plot without prediction  
# plt.plot(range(1, len(belief_history_no_prediction) + 1), belief_history_no_prediction, 
#          'r-s', linewidth=2, markersize=8, label='Measurement Only (No Prediction)')

# plt.xlabel('Measurement Step')
# plt.ylabel('Belief (Door Open)')
# plt.title('Belief Progression: With vs Without Prediction Steps')
# plt.legend()
# plt.grid(True)
# plt.ylim(0, 1)

# # Add measurement labels
# measurement_labels = ['open', 'open', 'open', 'open', 'open', 'closed', 'open', 'open']
# plt.xticks(range(1, len(belief_history_with_prediction) + 1), measurement_labels, rotation=45)
# plt.yticks(np.arange(0, 1.01, 0.05))  # Every 0.05 from 0 to 1
# plt.tight_layout()
# plt.show()