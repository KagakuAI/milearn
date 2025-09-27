import logging
import warnings


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        keywords = ["available:", "CUDA", "LOCAL_RANK:"]
        return not any(keyword in record.getMessage() for keyword in keywords)


warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(IgnorePLFilter())
logging.getLogger("pytorch_lightning.accelerators.cuda").addFilter(IgnorePLFilter())
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
