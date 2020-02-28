from src.models.model import Model

class Autoencoder(Model):
    """
    An autoencoder is a model split into an encoder and a decoder. The 
    attribute first_decoder_layer_name marks this division.
    """
    def __init__(self, config, in_channels, img_size):
        super().__init__(config=config, in_channels=in_channels, img_size=img_size)
        self.first_decoder_layer_name = None

    def encode(self, x):
        for layer_name, layer in self.layers.items():
            if layer_name == self.first_decoder_layer_name:
                break
            else:
                for operation in self.operations_before.get(layer_name, []):
                    x = operation(x)
                x = layer(x)
                for operation in self.operations_after.get(layer_name, []):
                    x = operation(x)
        return x

    def decode(self, x):
        decoder_reached = False
        for layer_name, layer in self.layers.items():
            if layer_name == self.first_decoder_layer_name:
                decoder_reached = True
            if decoder_reached:
                for operation in self.operations_before.get(layer_name, []):
                    x = operation(x)
                x = layer(x)
                for operation in self.operations_after.get(layer_name, []):
                    x = operation(x)
        return x