from pyfedappwrap.engine.runtime import FedDBEngine

from app import AgglomerativeClusteringAPP

engine = FedDBEngine()
engine.register(AgglomerativeClusteringAPP())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()