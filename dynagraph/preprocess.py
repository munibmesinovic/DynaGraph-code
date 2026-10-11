import argparse

import yaml

from .preprocessing import ehrshot, eicu, hirid, mimic

PIPELINES = {"mimic": mimic, "eicu": eicu, "hirid": hirid, "ehrshot": ehrshot}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(PIPELINES))
    p.add_argument("--config", required=True)
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    PIPELINES[a.dataset].run_pipeline(cfg)


if __name__ == "__main__":
    main()
