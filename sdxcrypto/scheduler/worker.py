from datetime import datetime

#import from this library
import firebase
from scheduler.inference import Inference
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
        logger.info(f'Setting worker status to {self.current_job}')

        #change job_status == in-process
        self.job_uuid = params['job_uuid']
        logger.info(f'Starting job for Job ID: {self.job_uuid}')
        self.change_job_status('in-process')

        #run_inference  
        self.inf.run_inference(params)

        #change job_status == done
        self.change_job_status('done')

        #change current_job ==False
        self.current_job = False
        logger.info(f'Setting worker status to {self.current_job}')

    def change_job_status(self, changeto):
        if changeto == 'in-process':
            toupdate = {'job_status': changeto,
                        'job_in_process': str(datetime.timestamp(datetime.now()))}
            firebase.update_job(self.job_uuid, toupdate)
        else:
            toupdate = {'job_status': changeto,
                        'job_done': str(datetime.timestamp(datetime.now()))}
            firebase.update_job(self.job_uuid, toupdate)
