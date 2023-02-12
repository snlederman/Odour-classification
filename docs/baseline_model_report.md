---
title: The Locust antenna as an odor discriminator
subtitle: Baseline model report
geometry: "left=2cm, right=2cm, top=0cm, bottom=0cm"
# pandoc -f markdown+hard_line_breaks --output docs/baseline_model_report.pdf docs/baseline_model_report.md
---

This is our baseline model, which only sample from the distribution of train data labels in order to predict the input features.

```python
class BaselineModel:
    """random sampler from train labels"""
    def __init__(self):
        self.labels = None
        
    def fit(self, X, y):
        """recives pandas series as training labels and store them for predictions"""
        self.labels = y.values

    def predict(self, X):
        """predict random labels from training set at the length of input"""
        res = np.random.choice(self.labels, X.shape[0])
        return res
```

This is the output of the model:
```
Report:
    Accuracy: 13%
    Precision: 14%
    Sensitivity: 12%
    F1 score: 13%
```