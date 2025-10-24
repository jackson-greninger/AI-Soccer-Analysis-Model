import matplotlib.pyplot as plt

player_counts = []

# inside loop
player_counts.append(sum(int(box.cls[0]) == 0 for box in results.boxes))

# after loop
plt.plot(player_counts)
plt.title("Number of Players Detected per Frame")
plt.xlabel("Frame")
plt.ylabel("Count")
plt.show()
