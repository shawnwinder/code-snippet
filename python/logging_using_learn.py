# logger test
logger = logging.getLogger()

# level
# logger.setLevel('INFO')
logger.setLevel('DEBUG')

# format
BASIC_FORMAT = "[%(asctime)s %(levelname)s %(pathname)s] %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

# handler
chlr = logging.StreamHandler() # console handler
chlr.setFormatter(formatter)
# chlr.setLevel('INFO') # can be omitted, if so, same as # logger's level
fhlr = logging.FileHandler('tmp.log') # file handler
fhlr.setFormatter(formatter)
# fhlr.setLevel('INFO')
logger.addHandler(chlr)
logger.addHandler(fhlr)

# logging
logger.info('this is info')
logger.debug('this is debug')

