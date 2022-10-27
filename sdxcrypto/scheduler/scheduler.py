
import sys
from sqlalchemy.orm import Session

#import from this lib
from db.db import engine
from models import models
from scheduler.worker import Worker
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')

def init_scheduler(queue):
    worker = Worker()
    logger.info('Worker extended and scheduler started')
    while True:
        try:
            if not worker.current_job:
                #get next job in queue
                job_id = queue.get()
                #get params of the job
                with Session(engine) as session:
                    params = session.query(models.Job).filter(models.Job.job_uuid == job_id).first()
                #run job
                worker.run_job(params)

            else:
                pass

        # except KeyboardInterrupt:
        #     logger.exception('Exiting scheduler')
        #     sys.exit(0)

        except Exception as e:
            logger.exception(e)
