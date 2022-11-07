
import sys

#import from this lib
import firebase
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
                job_uuid = queue.get()
                #get params of the job
                params = firebase.get_job(job_uuid)
                params['job_uuid'] = job_uuid

                logger.info(params)
                
                #run job
                worker.run_job(params)

            else:
                pass

        # except KeyboardInterrupt:
        #     logger.exception('Exiting scheduler')
        #     sys.exit(0)

        except Exception as e:
            logger.exception(e)
