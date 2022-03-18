import torch as t

class GeneralLinearFullNet(t.nn.Module):
    def __init__(
            self, in_features, out_features,
            hidden_sizes=[],
            nonlinearity=None,
            end_with_nonlinearity=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.sizes = [in_features] + hidden_sizes + [out_features]
        self.hidden_layers = len(hidden_sizes)
        self.layers = t.nn.ModuleList()
        self.end_with_nonlinearity = end_with_nonlinearity
        for i in range(len(self.sizes) - 1):
            self.layers.append(t.nn.Linear(self.sizes[i], self.sizes[i+1]))
            if i < len(self.sizes) - 2 or end_with_nonlinearity:
                self.layers.append(nonlinearity())

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
