#!/usr/bin/env python3
"""Report raw results and a separately documented whole-pair sleep recheck."""
import argparse
import datetime
import json
from pathlib import Path

from entropy_efficiency import analyze, distribution, require, sha, IMAGES
from repeat_entropy_pair import audit


def report(source, recheck, sleep_events):
    raw = analyze(source)
    repeated = audit(recheck)
    require(Path(repeated['source']) == source, 'Supplement names another source capture')
    original = json.loads((source / 'manifest.json').read_text())
    excluded = [r for r in original['commands'] if r['image'] == repeated['image'] and r['repetition'] == 2]
    require(len(excluded) == 2, 'Expected the complete interrupted second pair')
    events = []
    for line in sleep_events.read_text().splitlines():
        if 'Entering Sleep state' not in line: continue
        when = datetime.datetime.strptime(line[:25], '%Y-%m-%d %H:%M:%S %z')
        if any(datetime.datetime.fromisoformat(r['started_utc']) <= when <=
               datetime.datetime.fromisoformat(r['finished_utc']) for r in excluded):
            events.append(line)
    require(events, 'No recorded system sleep overlaps the replaced pair')
    images = []
    for image in IMAGES:
        routes = {}
        for route in ('exact', 'estimated'):
            rows = [r for r in original['commands'] if r['image'] == image and r['route'] == route
                    and not (image == repeated['image'] and r['repetition'] == 2)]
            labels = [str(source / (r['tag'] + '.stdout')) for r in rows]
            if image == repeated['image']:
                rows.append(next(r for r in repeated['commands'] if r['route'] == route))
                labels.append(str(recheck / (route + '.stdout')))
            require(len(rows) == 2, 'Uninterrupted comparison does not contain two complete pairs')
            routes[route] = dict(samples=labels, bytes=rows[0]['result']['encoded_bytes'],
                seconds=distribution([r['result']['seconds'] for r in rows]),
                stages={k: distribution([r['result']['stages'][k] for r in rows]) for k in rows[0]['result']['stages']},
                ans_cost_seconds=distribution([r['result']['entropy_work']['ans_token_cost_ns']/1e9 for r in rows]),
                peak_backing=max(r['result']['peak_backing'] for r in rows),
                planned_capacity=max(r['result']['planned_capacity'] for r in rows))
        images.append(dict(image=image, routes=routes,
            ratio=routes['exact']['seconds']['median']/routes['estimated']['seconds']['median'],
            extra_bytes=routes['estimated']['bytes']-routes['exact']['bytes']))
    totals = {route: dict(seconds=sum(row['routes'][route]['seconds']['median'] for row in images),
              bytes=sum(row['routes'][route]['bytes'] for row in images)) for route in ('exact', 'estimated')}
    return dict(schema=2, raw=raw, recheck=dict(path=str(recheck), manifest_sha256=sha(recheck/'manifest.json'),
        reason=repeated['reason'], excluded_pair=[r['tag'] for r in excluded],
        evidence=dict(path=str(sleep_events), sha256=sha(sleep_events), overlapping_events=events),
        selection='Replace the entire externally interrupted pair; preserve original raw matrix; no subtraction or fastest-run selection.'),
        qualification=dict(images=images, totals=totals,
            ratio=totals['exact']['seconds']/totals['estimated']['seconds'],
            extra_bytes=totals['estimated']['bytes']-totals['exact']['bytes']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--recheck', type=Path, required=True)
    p.add_argument('--sleep-events', type=Path, required=True)
    p.add_argument('--json', type=Path, required=True)
    args = p.parse_args()
    result = report(args.source.resolve(), args.recheck.resolve(), args.sleep_events.resolve())
    args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result['qualification'][k] for k in ('totals','ratio','extra_bytes')}, indent=2))
