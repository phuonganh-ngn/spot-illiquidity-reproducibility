"""Run twice with all Python network paths blocked, and compare deterministic files."""
import sys,json,subprocess,hashlib
from common import *
def hashes():
 result={}
 for folder in ['data/processed','outputs']:
  for f in sorted((ROOT/folder).rglob('*')):
   if f.is_file() and f.name not in ['offline_reproduction_certificate.json']:result[str(f.relative_to(ROOT))]=sha(f)
 return result

def main():
 for iteration in [1,2]:
  log=ROOT/'logs'/f'offline_rebuild_{iteration}.log'
  with log.open('w') as stream:subprocess.run([sys.executable,str(ROOT/'src/run_pipeline.py'),'--offline'],cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,check=True)
  after=hashes()
  if iteration==1:before=after
  else:
   changed=[k for k in set(before)|set(after) if before.get(k)!=after.get(k)]
   assert not changed,'OFFLINE_REPRODUCTION_FAILED: '+str(changed)
 cert=dict(status='PASS',network_disabled='Python requests/session, DNS and socket connect layers raise before access; no network subprocesses in builder',deterministic_outputs_checked=len(after),comparison='Two independent full offline builds; byte-identical data and audit outputs',source_code_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').glob('*.py'))},raw_inventory_sha256=sha(ROOT/'outputs/raw_hash_inventory.csv'))
 (ROOT/'outputs/offline_reproduction_certificate.json').write_text(json.dumps(cert,sort_keys=True,indent=2)+'\n')
 from finalize_reports import main as finalize
 finalize(reproduced=True);final_before=hashes();finalize(reproduced=True);assert hashes()==final_before,'Final report reproduction failed'
 state=json.loads((ROOT/'outputs/readiness_summary.json').read_text());assert state['ready_for_modeling']=='YES' and not state['fatal_blockers'],'Readiness certification is internally inconsistent'
 print('OFFLINE REPRODUCTION: PASS;',len(after),'byte-identical outputs; final reports deterministic')
if __name__=='__main__':main()
