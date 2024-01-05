from recognizer.raw_samples import *
from IPython.display import clear_output
import time
import os

def write_trims_for_suite(suite_path: str, overwrite=False):
    suite = load_raw_suite(suite_path)
    for sample in suite.samples:
        if not overwrite and os.path.isfile(os.path.join(sample.source, "trim.txt")):
            continue

        max_interval = int(sample.getLength())
        interval = 10

        is_interval_ok = max_interval > interval

        if not is_interval_ok:
            print("Interval too big, setting to sample length")
            interval = max_interval

        ok = False
        while not ok:
            start_ok = False
            end_ok = False
            clear_output(wait=True)
            print(f"Sample length: {max_interval}")
            print(f"Showing {sample.source} with interval {interval}")
            sample.graph_edges(interval)
            time.sleep(1)
            start_trim = input("Start trim")
            if start_trim == "skip":
                break
            elif start_trim == "end":
                return
            elif start_trim == "more":
                interval += 5
                if not is_interval_ok:
                    interval = max_interval
                print(interval)
                continue
            else:
                try:
                    start = int(start_trim)
                    start_ok = True
                except ValueError:
                    print("ValueError")
                    continue


            end_trim = input("End trim")
            if end_trim == "skip":
                break
            elif end_trim == "end":
                return
            elif end_trim == "more":
                interval += 5
                if not is_interval_ok:
                    interval = max_interval
                continue
            else:
                try:
                    end = int(end_trim)
                    end_ok = True
                except ValueError:
                    print("ValueError")
                    continue

            if start_ok and end_ok:
                ok = True

        if ok:
            sample.save_trim(start, end)
        else:
            continue
