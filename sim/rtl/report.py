"""Parse Vivado metrics conservatively; missing results remain null."""
import argparse,json,re
from pathlib import Path

def parse(directory):
    p=Path(directory)
    def read(name): return (p/name).read_text(errors='replace') if (p/name).exists() else ''
    util=read('utilization_route.rpt'); timing=read('timing_route.rpt')
    values={}
    for label in ('Slice LUTs','CLB LUTs','LUT as Logic','Slice Registers','CLB Registers','Block RAM Tile','RAMB36/FIFO','RAMB18','DSPs'):
        m=re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d,.]+)',util)
        values[label]=float(m[1].replace(',','')) if m else None
    m=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+.*?\n\s*([-\d.]+)\s+([-\d.]+)\s+\d+\s+\d+\s+([-\d.]+)\s+([-\d.]+)',timing,re.S)
    return dict(resources=values,wns_ns=float(m[1]) if m else None,tns_ns=float(m[2]) if m else None,whs_ns=float(m[3]) if m else None,ths_ns=float(m[4]) if m else None,route_report_present=bool(read('route_status.rpt')),drc_report_present=bool(read('drc_route.rpt')),metadata=read('metadata.txt'),manifest=json.loads(read('manifest.json') or '{}'),verified_implementation=False,note='Review route completion, DRC, unconstrained endpoints and setup/hold before declaring a passing implementation. No frequency inferred from WNS. Peak RAM from Vivado log must be attached.')
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('directory'); args=a.parse_args(); print(json.dumps(parse(args.directory),indent=2))
