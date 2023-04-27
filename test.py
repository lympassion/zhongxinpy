import json
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
pred_np = arr
pred_dict = {}
for i in range(5):
    pred_dict[str(i)] = int(pred_np[i])
pred_json = json.dumps(pred_dict)
print(pred_json)

