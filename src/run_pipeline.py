"""Online discovery or strictly network-disabled reconstruction. No modeling."""
import argparse,sys,json,socket
from pathlib import Path

def disable_network():
 def forbidden(*args,**kwargs):raise RuntimeError('OFFLINE_NETWORK_ACCESS_FORBIDDEN')
 socket.create_connection=forbidden;socket.getaddrinfo=forbidden;socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
 import requests
 requests.sessions.Session.request=forbidden

def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--online',action='store_true');group.add_argument('--offline',action='store_true');args=parser.parse_args();offline=not args.online
 if offline:disable_network()
 import download_data,probe_semantics,audit_raw_data,validate_semantics,detect_metric_layout,build_daily_features,build_future_outcomes,report_audit,verify_dataset,finalize_reports
 from common import ROOT,sha
 saved=sys.argv;sys.argv=[saved[0]]+(['--offline'] if offline else [])
 download_data.main();sys.argv=saved
 probe_semantics.main(offline)
 audit_raw_data.main();validate_semantics.main();detect_metric_layout.main();build_daily_features.main();build_future_outcomes.main();report_audit.main();verify_dataset.main()
 cert=ROOT/'outputs/offline_reproduction_certificate.json';trusted=False
 if cert.exists():
  record=json.loads(cert.read_text());current={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').glob('*.py'))};trusted=record.get('status')=='PASS' and record.get('source_code_sha256')==current
 finalize_reports.main(reproduced=trusted)
if __name__=='__main__':main()
