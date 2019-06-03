import logging

logging.basicConfig(format='%(asctime)s:%(message)s', datafmt='%Y/%m/%d %I-%M-%S %p', level=logging.DEBUG)
logging.debug("This message should go to console")
logging.info("so should this")
logging.warning("And this")
