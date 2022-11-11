from multiprocessing import Process, Queue, set_start_method
import os

#import from this library
from scheduler import scheduler
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')

global scheduler_queue
scheduler_queue = None


def init_scheduler():
    logger.info('Starting scheduler...')

    global scheduler_queue
    if scheduler_queue is None:
        scheduler_queue = Queue()
        Process(target=scheduler.init_scheduler, args=(scheduler_queue, )).start()
 
def schedule_job(uuid):
    global scheduler_queue
    scheduler_queue.put(uuid)