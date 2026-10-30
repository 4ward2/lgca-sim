"""Validated V=2 tables to spatial, synchronous SystemVerilog."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sim.cert import tables
VERSION = '1.0'

def checked(t, task):
    tables.validate(t, task)
    if set(t) - {'dims','c','V','u','r','o','checkpoint_sha256','table_sha256'}:
        raise ValueError('unknown table fields')
    if t['V'] != 2: raise ValueError('only V=2 supported')
    digest = hashlib.sha256(tables.hashable(t).encode()).hexdigest()
    if t.get('table_sha256', digest) != digest: raise ValueError('canonical hash mismatch')
    return digest

def generate(t, task, shape, dest, source_sha=None, period=10):
    digest = checked(t, task)
    if len(shape) != t['dims'] or any(type(n) is not int or n < 1 for n in shape): raise ValueError('bad shape')
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    n = int(np.prod(shape)); w = t['c']+1
    lines = [f'''module core(input logic clk,rst,start,in_valid,out_ready,
 output logic in_ready,out_valid,out_bit,busy,done,
 input logic [{w-1}:0] in_code, output logic [1:0] status,
 output integer rounds,total_cycles);
 localparam integer N={n}, LIMIT={2*n+1};
 logic [{w-1}:0] codes[0:N-1];
 logic [1:0] domains[0:N-1], nxt[0:N-1];
 integer loaded, emitted, computes;
 logic equal_all;
 assign in_ready = !busy && loaded<N;
 assign out_valid = done && status==1 && emitted<N;
 function automatic [1:0] unary(input integer x);
 case(x)''']
    for x,u in enumerate(t['u']): lines.append(f'{x}: unary=2\'d{sum(v<<h for h,v in enumerate(u))};')
    lines += ['default: unary=0; endcase endfunction', 'function automatic bit rel(input integer e,a,b,j,h); case (e)']
    for e,axis in enumerate(t['r']):
        lines.append(f'{e}: case(a)')
        for a,row in enumerate(axis):
            lines.append(f'{a}: case(b)')
            for b,R in enumerate(row):
                bits=sum(R[j][h] << (2*j+h) for j in range(2) for h in range(2))
                lines.append(f"{b}: rel=(4'd{bits} >> (2*j+h)) & 1;")
            lines.append('default: rel=0; endcase')
        lines.append('default: rel=0; endcase')
    lines += ['default: rel=0; endcase endfunction', '''function automatic [1:0] support(input integer e,a,b,input logic [1:0] d,input bit right_side);
 integer h,j; begin support=0; for(h=0;h<2;h=h+1) for(j=0;j<2;j=j+1)
 support[h]=support[h] | (d[j] & (right_side ? rel(e,a,b,j,h) : rel(e,a,b,h,j)));
 end endfunction
 function automatic bit readout(input integer x,d); case(x)''']
    for x,row in enumerate(t['o']):
        bits=sum(v<<i for i,v in enumerate(row)); lines.append(f"{x}: readout=(4'd{bits} >> d) & 1;")
    lines += ['default: readout=0; endcase endfunction', 'always_comb begin', 'equal_all=1;']
    for i,p in enumerate(np.ndindex(tuple(shape))):
        expr=f'unary(codes[{i}])'
        for e in range(len(shape)):
            for delta in (-1,1):
                q=list(p); q[e]+=delta
                if 0<=q[e]<shape[e]:
                    k=int(np.ravel_multi_index(tuple(q),shape)); y=f'codes[{k}]'; d=f'domains[{k}]'
                else: y='0'; d="2'd3"
                a,b=(y,f'codes[{i}]') if delta<0 else (f'codes[{i}]',y)
                expr+=f' & support({e},{a},{b},{d},1\'b{int(delta<0)})'
        lines += [f'nxt[{i}]={expr};', f'if(nxt[{i}]!=domains[{i}]) equal_all=0;']
    lines += ['end', '''always_comb begin out_bit=0; if(emitted<N) out_bit=readout(codes[emitted],domains[emitted]); end
 integer k;
 always_ff @(posedge clk) begin
 if(rst) begin loaded<=0; emitted<=0; computes<=0; rounds<=0; total_cycles<=0; busy<=0; done<=0; status<=0;
 for(k=0;k<N;k=k+1) begin domains[k]<=3; codes[k]<=0; end
 end else begin
 if(in_valid && in_ready) begin codes[loaded]<=in_code; loaded<=loaded+1; end
 if(start && !busy && loaded==N && !out_valid) begin
 busy<=1; done<=0; status<=0; rounds<=0; computes<=0; emitted<=0; total_cycles<=1;
 for(k=0;k<N;k=k+1) domains[k]<=3;
 end else if(busy) begin
 total_cycles<=total_cycles+1; computes<=computes+1;
 for(k=0;k<N;k=k+1) domains[k]<=nxt[k];
 if(equal_all) begin busy<=0; done<=1; status<=1; end
 else if(computes+1>=LIMIT) begin busy<=0; done<=1; status<=2; end
 else rounds<=rounds+1;
 end else if(out_valid) begin
 total_cycles<=total_cycles+1;
 if(out_ready) begin emitted<=emitted+1; if(emitted==N-1) loaded<=0; end
 end
 end end
 endmodule''']
    (dest/'core.sv').write_text('\n'.join(lines))
    manifest=dict(generator_version=VERSION,table_sha256=digest,input_sha256=source_sha,task=task,shape=shape,code_width=w,period_ns=period,architecture='spatial synchronous; one compute comparison per clock',interface=dict(load='row-major; N handshakes before a later start edge; no simultaneous final load/start',start='idle, fully loaded, no pending output; repeated busy start ignored',status='0 idle/running, 1 fixed point, 2 timeout; done sticky until start/reset',rounds='strict changes; computes=rounds+1 on success',total_cycles='start + compute + output clocks including backpressure; load clocks counted externally',reuse='drain output then load next input; domains held until next start',reset='synchronous active high; overrides all handshakes'))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('table'); p.add_argument('--task',required=True,choices=['P','A','M']); p.add_argument('--shape',type=int,nargs='+',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    raw=Path(a.table).read_bytes(); generate(json.loads(raw),a.task,a.shape,a.out,hashlib.sha256(raw).hexdigest())
if __name__=='__main__': main()
