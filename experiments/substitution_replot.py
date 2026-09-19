"""Regenerate report/figures only from saved simulation data."""
import json
import hashlib
from pathlib import Path
from experiments.substitution_complementarity import report
from experiments.substitution_readout import run

if __name__=='__main__':
    out=Path('outputs/substitution_complementarity_v1')
    config=Path('configs/substitution_complementarity.json')
    report(out,json.loads(config.read_text(encoding='utf-8')))
    run(out)
    sources=['experiments/substitution_complementarity.py','experiments/substitution_readout.py',__file__]
    (out/'report_verification.json').write_text(json.dumps({
        'source_sha256':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        'simulation_rerun':False},indent=2),encoding='utf-8')
