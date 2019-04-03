list = []
list.append(1)
list.append(2)
list.append(3)

GAMMA = 0.3
discounted_factor = 1

gt = []
for i, reward in enumerate(reversed(list)):
    if i == 0:
        gt.append(reward)
    else:
        gt.append(reward + GAMMA * gt[-1])
print(gt)