from tensorflow.keras.applications import vgg16, vgg19, resnet50, resnet, \
    densenet, mobilenet, mobilenet_v2


class FeatureExtractor(object):
    def __init__(self,
                 feature_extractor_id,
                 input_shape,
                 weights,
                 output_pooling,
                 **kwargs
                 ):
        self.feature_extractor_id = feature_extractor_id
        self.input_shape = input_shape
        self.weights = weights
        self.output_pooling = output_pooling

    def get_vgg16_extractor(self):
        return vgg16.VGG16(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_vgg19_extractor(self):
        return vgg19.VGG19(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_resnet50_extractor(self):
        return resnet50.ResNet50(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_resnet101_extractor(self):
        return resnet.ResNet101(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_resnet152_extractor(self):
        return resnet.ResNet152(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_densenet121_extractor(self):
        return densenet.DenseNet121(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_densenet169_extractor(self):
        return densenet.DenseNet169(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_densenet201_extractor(self):
        return densenet.DenseNet201(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_mobilenetv1_extractor(self):
        return mobilenet.MobileNet(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_mobilenetv2_extractor(self):
        return mobilenet_v2.MobileNetV2(
            include_top=False,
            input_shape=self.input_shape,
            weights=self.weights,
            pooling=self.output_pooling
        )

    def get_feature_extractor(self):
        if self.feature_extractor_id == "vgg16":
            feature_extractor = self.get_vgg16_extractor()
        elif self.feature_extractor_id == "vgg19":
            feature_extractor = self.get_vgg19_extractor()
        elif self.feature_extractor_id == "resnet50":
            feature_extractor = self.get_resnet50_extractor()
        elif self.feature_extractor_id == "resnet101":
            feature_extractor = self.get_resnet101_extractor()
        elif self.feature_extractor_id == "resnet152":
            feature_extractor = self.get_resnet152_extractor()
        elif self.feature_extractor_id == "densenet121":
            feature_extractor = self.get_densenet121_extractor()
        elif self.feature_extractor_id == "densenet169":
            feature_extractor = self.get_densenet169_extractor()
        elif self.feature_extractor_id == "densenet201":
            feature_extractor = self.get_densenet201_extractor()
        elif self.feature_extractor_id == "mobilenetv1":
            feature_extractor = self.get_mobilenetv1_extractor()
        elif self.feature_extractor_id == "mobilenetv2":
            feature_extractor = self.get_mobilenetv2_extractor()
        else:
            assert False, f"feature_extractor_id {self.feature_extractor_id} not found."
        return feature_extractor
