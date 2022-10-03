# a helper function to keep track of base models, models owned by the individuals, images/assets generated and owned by the individual
import pandas as pd 

# d = pd.DataFrame(columns=['model_type', 'model_name', 'model_dir'])
# d.loc[0] = ['base_model', 'CompVis/stable-diffusion-v1-4', ""]
# d.to_csv('storage/data.csv')
# print(d)

class BaseModels:
    def base_models(self):
        data = pd.read_csv("storage/data.csv")
        bm = list(data["model_name"])
        return bm