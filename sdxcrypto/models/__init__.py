# from db.db import get_db
# from models import models

# #add base_models
# db = get_db()
# base_model = ["CompVis/stable-diffusion-v1-4"]
# for bm in base_model:
#     toadd = models.BaseModels({'name':bm})
#     db.add(toadd)
#     db.commit()
#     db.refresh(toadd)