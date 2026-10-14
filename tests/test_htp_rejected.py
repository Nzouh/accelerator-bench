import json
import sys
from types import SimpleNamespace

from accelerator_bench import htp


def test_rejected_diagnostics_are_opt_in_and_keep_rejection(tmp_path, monkeypatch):
    result = {'records': [{'case_id': 'bad', 'variant': 'optimized', 'job_id': 'job',
                           'key': 'bad_r1', 'repeat': 1, 'status': 'rejected'}]}
    (tmp_path/'results.json').write_text(json.dumps(result))
    folder = tmp_path/'htp'/'bad_r1'
    folder.mkdir(parents=True)
    (folder/'job_qhas_summary.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'qai_hub', SimpleNamespace(set_verbose=lambda x: None))
    monkeypatch.setattr(htp, 'summarize_htp', lambda raw: {'htp_nodes': 5})
    assert htp.collect_htp(tmp_path) == []
    rows = htp.collect_htp(tmp_path, include_rejected=True)
    assert rows[0]['measurement_status'] == 'rejected'
    assert json.loads((tmp_path/'results.json').read_text()) == result
