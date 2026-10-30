"""Run real XSim against all domain bits at every compute edge."""
import argparse,itertools,json,subprocess,time,ctypes
from pathlib import Path
import numpy as np
from sim.cert import tables
from sim.rtl.gen_sv import generate
from sim.rtl.reference_trace import trace
ROOT=Path(__file__).resolve().parents[2]
BIN=Path('C:/AMDDesignTools/2026.1/Vivado/bin')

def free_ram():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    m=Memory(); m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError('RAM query failed')
    return m.avail_phys

def run(cmd,cwd):
    if free_ram()<3*1024**3: raise RuntimeError('resource exit: free RAM below 3 GiB; use cycle_model')
    p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,timeout=240)
    with (cwd/'tool.log').open('a') as f: f.write(p.stdout+p.stderr)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    return p.stdout+p.stderr

def check_design(t,task,shape,cases,name):
    dest=ROOT/'sim/rtl/build'/name; generate(t,task,list(shape),dest)
    n=int(np.prod(shape)); w=t['c']+1; s=[f'''module tb;
 logic clk=0; always #5 clk=~clk;
 logic rst=1,start=0,in_valid=0,out_ready=0; logic [{w-1}:0] in_code=0;
 wire in_ready,out_valid,out_bit,busy,done; wire [1:0] status; integer rounds,total_cycles;
 core dut(.*);
 task tick; begin @(posedge clk); #1; end endtask
 task fail(input integer tag); begin $display("FAIL %0d",tag); $fatal; end endtask
 initial begin tick; @(negedge clk); rst=0;
''']; comparisons=0; cycles=[]
    for ci,xi in enumerate(cases):
        states,out,r=trace(t,xi); cycles.append(dict(rounds=r,compute=r+1,load=n,minimum_total=2*n+r+2))
        for x in xi.flat:
            s.append(f'@(negedge clk); in_valid=1; in_code={int(x)}; if(!in_ready) fail(1); tick;')
        s.append('@(negedge clk); in_valid=0; start=1; tick; if(!busy || done || rounds!=0 || total_cycles!=1) fail(2);')
        for j,st in enumerate(states):
            if j:
                # Repeated start and spurious input while busy must be ignored.
                s.append('@(negedge clk); start=1; in_valid=1; tick;')
                expected_done=int(j==len(states)-1)
                s.append(f'if(done!={expected_done} || busy!={1-expected_done} || rounds!={min(j,r)} || total_cycles!={j+1} || in_ready) fail(3);')
            for k,d in enumerate(st.flat): s.append(f"if(dut.domains[{k}] !== 2'd{int(d)}) fail({1000+ci});")
            comparisons+=n
        s.append('@(negedge clk); start=0; in_valid=0; out_ready=0; tick;')
        for k,bit in enumerate(out.flat):
            s.append(f'if(!out_valid || out_bit!==1\'b{int(bit)} || status!=1) fail(4);')
            s.append('@(negedge clk); out_ready=0; tick;')
            s.append(f'if(!out_valid || out_bit!==1\'b{int(bit)} || !done) fail(5);')
            s.append('@(negedge clk); out_ready=1; tick;')
        s.append('@(negedge clk); out_ready=0; if(out_valid || !in_ready || !done) fail(6);')
    s+=['rst=1; tick; if(done || busy || out_valid || rounds || total_cycles || !in_ready) fail(7);',f'$display("COSIM_PASS cases={len(cases)} domains={comparisons}"); $finish; end endmodule']
    (dest/'tb.sv').write_text('\n'.join(s))
    run([str(BIN/'xvlog.bat'),'-sv','core.sv','tb.sv'],dest)
    run([str(BIN/'xelab.bat'),'tb','-s','test_sim','-mt','off'],dest)
    output=run([str(BIN/'xsim.bat'),'test_sim','-runall'],dest)
    if 'COSIM_PASS' not in output or 'FAIL ' in output: raise AssertionError(output)
    return dict(name=name,cases=len(cases),domain_comparisons=comparisons,cycles=cycles)

def oracle(task):
    c=1 if task=='P' else 2; dims=2 if task=='M' else 1; nin=1<<(c+1)
    # Independent equality relation; unary includes empty, singleton, and full.
    return dict(dims=dims,c=c,V=2,u=[[0,0],[1,0],[0,1],[1,1]]*(nin//4),r=[[[[[int(j==h) for h in range(2)] for j in range(2)] for b in range(nin)] for a in range(nin)] for e in range(dims)],o=[[0,1,1,1] for x in range(nin)])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); a=p.parse_args()
    rng=np.random.default_rng(110801); results=[]; begin=time.time()
    for path in sorted((ROOT/'sim/cert/fixtures').glob('*_tables.json')):
        t=json.loads(path.read_text()); task=path.name[0]; shapes=[(1,),(2,),(4,),(16,)] if task!='M' else [(1,1),(2,2),(3,3),(8,8)]
        if a.quick: shapes=shapes[:1]
        for shape in shapes:
            cases=[rng.integers(0,1<<(t['c']+1),shape) for _ in range(4)]
            cases += [np.full(shape,x,dtype=int) for x in (0,1,(1<<(t['c']+1))-1)]
            # Long carry/parity, walls and multiple sources emerge from code patterns;
            # additionally test every valid code independently on short chains.
            if int(np.prod(shape))<=4 and task!='M':
                cases += [np.array(x).reshape(shape) for x in itertools.product(range(1<<(t['c']+1)),repeat=shape[0])]
            for xi in cases[:4]: tables.confirm(t,xi)
            result=check_design(t,task,shape,cases,path.stem+'_'+'x'.join(map(str,shape)))
            results.append(result); (ROOT/'sim/rtl/build/partial_results.json').write_text(json.dumps(results,indent=2)); print(result['name'],result['cases'],flush=True)
    for task in ('P','A','M'):
        t=oracle(task); shape=(4,) if task!='M' else (2,2)
        cases=[np.array(x).reshape(shape) for x in itertools.product(range(4),repeat=4)]
        results.append(check_design(t,task,shape,cases,'oracle_'+task))
    summary=dict(seed=110801,simulator='Vivado XSim 2026.1',elapsed_seconds=time.time()-begin,designs=len(results),cases=sum(x['cases'] for x in results),domain_comparisons=sum(x['domain_comparisons'] for x in results),results=results)
    (ROOT/'sim/rtl/build/results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
if __name__=='__main__': main()

