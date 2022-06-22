# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path


custom_scenes = [
#    "lego"
]

llff_scenes = [
#    "fern",
#    "flower",
#    "fortress",
#    "horns",
#    "leaves",
#    "orchids",
#    "room",
    "trex"
]

tnt_scenes = [
#    "Family",
#    "Horse",
#    "M60",
#    "Playground",
#    "Train",
#    "Truck"
]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Style image"),
        scene: str = Input(description="Scene", default='trex', choices=llff_scenes + tnt_scenes + custom_scenes),
    ) -> Path:
        """Run a single prediction on the model"""
        #if scene in llff_scenes:
            

        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
