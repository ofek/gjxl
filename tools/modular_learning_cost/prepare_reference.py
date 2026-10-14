#!/usr/bin/env python3
"""Add timers to the hash-bound, byte-qualified libjxl attribution overrides."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--previous-data', type=Path, required=True)
p.add_argument('--previous-tools', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
hashes = {}

def read(path):
    hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.read_text()

def change(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)

for path in ('lib/jxl/enc_modular.cc', 'lib/jxl/modular/encoding/enc_encoding.cc',
             'lib/jxl/modular/encoding/enc_ma.cc'):
    s = read(a.previous_data / 'overrides' / path)
    s = '#include "reference_profile.h"\n' + s
    if path.endswith('enc_ma.cc'):
        s = change(s, '  // TODO(veluca): take into account that different contexts can have different',
                   '  learning_reference_study::Timer study_timer("fit_time");\n'
                   '  // TODO(veluca): take into account that different contexts can have different')
    if path.endswith('enc_encoding.cc'):
        s = change(s, '  TreeSamples tree_samples;\n',
                   '  learning_reference_study::Timer study_timer("learning_time");\n'
                   '  TreeSamples tree_samples;\n')
        s = change(s, 'tree_samples.SetPredictor(options[start].predictor,',
                   'tree_samples.SetPredictor(modular_study::Setting("STUDY_ALL_PREDICTORS") '
                   '? Predictor::Variable : options[start].predictor,')
    out = a.output / 'overrides' / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)

(a.output / 'overrides/study.h').write_text(read(a.previous_tools / 'study.h'))
(a.output / 'reference_effort.cpp').write_text(read(a.previous_data / 'reference_effort.cpp'))
(a.output / 'reference_runner.cpp').write_text(read(a.previous_tools / 'runner.cpp'))
(a.output / 'provenance.json').write_text(json.dumps(hashes, indent=2)+'\n')
