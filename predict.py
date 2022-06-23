# Hack to build CUDA extensions
import subprocess, uuid, os, sys
from datetime import datetime
subprocess.call(['pip', 'install', '/src'])

from cog import BasePredictor, Input, Path



custom_scenes = [
#    "lego"
]

llff_scenes = [
#    "fern",
    "flower",
#    "fortress",
#    "horns",
#    "leaves",
#    "orchids",
#    "room",
#    "trex"
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
        scene: str = Input(description="Scene", default='flower', choices=llff_scenes + tnt_scenes + custom_scenes),
    ) -> Path:
        """Run a single prediction on the model"""
        if scene in llff_scenes:
            data_type="llff"
            config_file=f'/src/opt/configs/llff_fixgeom.json'
        elif scene in tnt_scenes:
            data_type="tnt"
        elif scene in custom_scenes:
            data_type="custom"

        id = str(uuid.uuid4())

        ckpt_svox2=f'/root/.cache/arf-svox2/ckpt_svox2/{data_type}/{scene}'
        ckpt_arf=f'/root/.cache/arf-svox2/ckpt_arf/{data_type}/{scene}/{id}'
        data_dir=f'/root/.cache/arf-svox2/data/{data_type}/{scene}'
        
        ckpt_file=f'{ckpt_svox2}/ckpt.npz'        

        if not os.path.exists(ckpt_file):
            if not os.path.exists(ckpt_svox2):
                os.makedirs(ckpt_svox2)
            print('Downloading checkpoint for ' + scene, flush=True)
            subprocess.call(['wget', '--output-document', ckpt_file, '--quiet', f'https://arf.nmb.ai/ckpt_svox2/{data_type}/{scene}/ckpt.npz'])
            
        subprocess.call(
            ['python', 'opt_style.py', 
            '-t', ckpt_arf, data_dir, 
            '-c', config_file, 
            '--init_ckpt', ckpt_file,
            '--style', image,
            '--mse_num_epoches', '2',
            '--nnfm_num_epoches', '10',
            '--content_weight', '1e-3'            
            ], cwd='/src/opt'
        )

        subprocess.call(
            ['python', 'render_imgs.py', f'{ckpt_arf}/ckpt.npz', data_dir, '--render_path', '--no_imsave'], cwd='/src/opt'
        )

        return Path(f'{ckpt_arf}/test_renders_path.mp4')       
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
