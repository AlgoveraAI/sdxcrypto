# a helper function to keep track of base models, models owned by the individuals, images/assets generated and owned by the individual
import os
import pandas as pd 

class BaseModels:
    def __init__(self):
        self.cwd = os.getcwd()
        self.make_storage()

    def base_models(self):
        data = pd.read_csv(f"{self.cwd}/storage/data.csv")
        bm = list(data["model_name"])
        return bm

    def make_storage(self):
        if not os.path.exists(f"{self.cwd}/storage"):
            os.mkdir(f"{self.cwd}/storage")

        if not os.path.exists(f"{self.cwd}/storage/data.csv"):
            data = pd.DataFrame(columns=['model_type', 'model_name', 'model_dir'])
            data.loc[0] = ['base_model', 'CompVis/stable-diffusion-v1-4', ""]
            data.to_csv(f'{self.cwd}/storage/data.csv', index=False)

