from nst.datamodules.coco128_datamodule import COCO128DataModule


class TestCOCO128Datamodule:
    def test_coco_datamodule_should_initialize_without_error(self):
        datamodule = COCO128DataModule()
