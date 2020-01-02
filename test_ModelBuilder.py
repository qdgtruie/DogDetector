from ModelBuilder import ModelBuilder as Builder
import ImageSize

class TestModelBuilder:

    def test_build(self):
        net = Builder()
        model = net.build(width=ImageSize.WIDTH, height=ImageSize.HEIGHT, depth=3, classes=2)
        assert 11 == len(model.layers)