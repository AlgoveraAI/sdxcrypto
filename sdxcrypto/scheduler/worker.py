from sqlalchemy.orm import Session

#import from this library
from scheduler.inference import Inference
from models import models
from db.db import engine
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')


class Worker():
    def __init__(self):
        self.current_job = False
        self.job_uuid = None
        self.inf = Inference()

    def run_job(self, params):
        #set current_job == True
        self.current_job = True
        logger.info(f'Setting worker status to{self.current_job}')

        #change job_status == in-process
        self.job_uuid = params.job_uuid
        logger.info(f'Starting job for Job ID: {self.job_uuid}')
        self.change_job_status('in-process')

        #run_inference  
        self.inf.run_inference(params)

        #change job_status == done
        self.change_job_status('done')

        #change current_job ==False
        self.current_job = False
        logger.info(f'Setting worker status to{self.current_job}')

    def change_job_status(self, changeto):
        with Session(engine) as session:
            job_query = session.query(models.Jobs).filter(models.Jobs.job_uuid == self.job_uuid).first()
            job_query.job_status = changeto
            session.commit()