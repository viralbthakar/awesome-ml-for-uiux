from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from feature_extractor import FeatureExtractor


class ModelBuilder(object):
    def __init__(self, input_shape, output_shape, model_config, weights=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        self.model_config = model_config

    def create_feature_extractor(self, feature_extractor_id="vgg16", global_pooling_type=None):
        feature_extractor = FeatureExtractor(
            feature_extractor_id=feature_extractor_id,
            output_pooling=global_pooling_type,
            input_shape=self.input_shape,
            weights=self.weights).get_feature_extractor()
        return feature_extractor

    def create_prediction_head(self, layers):
        layers_to_add = []
        for layer in layers:
            layers_to_add.append(
                Dense(layer["num_nodes"],
                      activation=layer["activation"])
            )
        return Sequential(layers_to_add, name="prediction_head")

    def create_model_architecture(self):
        feature_extractor = self.create_feature_extractor(
            self.model_config["feature_extractor_id"],
            self.model_config["global_pooling_type"])

        prediction_head = self.create_prediction_head(
            self.model_config["layers"])

        input_image = Input(shape=self.input_shape, name="input_image")
        cnn_feature = feature_extractor(input_image)
        out_feature = prediction_head(cnn_feature)
        out = Dense(self.output_shape,
                    activation=self.model_config["output_activation"],
                    name="output_class")(out_feature)

        self.model = Model(inputs=input_image, outputs=out)
        print(self.model.summary())
        return self.model
