"""Actual optional Python callback and OS-boundary proof on synthetic inputs only."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from pathlib import Path
from typing import Any

from servegauge.comparison.isolation import ToolRuntime, prepare_process
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, Timings
from servegauge.mock.sealed_replay import ReplayAddress, ReplayObservation
from servegauge.runner.ledger import Ledger
from servegauge.storage.comparison import _guard
from servegauge.transport.http import assemble_stream
from servegauge.transport.sse import SseParser


def exercise_promptfoo_contract(directory: Path, runtime: ToolRuntime) -> dict[str, Any]:
    _guard(directory)
    if directory.exists():
        raise ValueError("fresh_external_contract_directory_required")
    directory.mkdir(parents=True)
    process = prepare_process(runtime, directory)
    fake_account = directory.parent / (directory.name + "-account-canary.txt")
    with fake_account.open("x") as stream:
        stream.write("SYNTHETIC_ACCOUNT_FILE_MUST_BE_UNREADABLE")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            accepted, _ = listener.accept()
            accepted.close()
        python_probe = """
import errno,json,os,socket,sys
from pathlib import Path
assert os.environ.get('SYNTHETIC_ACCOUNT_CANARY') is None
names=['OPENROUTER_API_KEY','SERVEGAUGE_OWNER_RUN','TRUEWEIGHTS_OWNER_RUN',
       'HTTP_PROXY','HTTPS_PROXY','NODE_OPTIONS','HOME']
assert not any(os.environ.get(k) for k in names)
refused=[]
channels=[(socket.AF_INET,socket.SOCK_STREAM,('127.0.0.1',int(sys.argv[1]))),
          (socket.AF_INET,socket.SOCK_STREAM,('192.0.2.1',443)),
          (socket.AF_INET,socket.SOCK_DGRAM,('192.0.2.1',443))]
for family,kind,target in channels:
    channel=socket.socket(family,kind)
    channel.settimeout(1)
    try:
        if kind==socket.SOCK_DGRAM:channel.sendto(b'synthetic-boundary-probe',target)
        else:channel.connect(target)
    except OSError as error:refused.append(error.errno in {errno.EPERM,errno.EACCES})
    finally:channel.close()
try:Path(sys.argv[2]).read_bytes()
except OSError as error:account=error.errno in {errno.EPERM,errno.EACCES}
else:account=False
assert len(refused)==3 and all(refused) and account
print(json.dumps({'tcp_udp_denied':3,'account_file_denied':True,'inherited_environment_absent':True}))
"""
        python = subprocess.run(
            process.wrap(runtime.python, ["-c", python_probe, str(port), str(fake_account)]),
            cwd=directory,
            env=process.environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        (directory / "python-boundary.log").write_text(python.stdout + python.stderr)
        if python.returncode:
            raise AssertionError("actual_python_boundary_probe_failed")
        python_result = json.loads(python.stdout)
        node_probe = r"""
const fs=require('node:fs'),net=require('node:net'),dgram=require('node:dgram');
(async()=>{
 if(process.env.SYNTHETIC_ACCOUNT_CANARY || process.env.HOME || process.env.OPENROUTER_API_KEY)
   throw new Error('inherited_environment');
 const denied=await Promise.all(['127.0.0.1','192.0.2.1'].map(host=>new Promise(resolve=>{
  const s=net.connect({host,port:host==='127.0.0.1'?Number(process.argv[1]):443});
  s.on('connect',()=>{s.destroy();resolve(false)});
  s.on('error',e=>{s.destroy();resolve(['EPERM','EACCES'].includes(e.code))});
  s.setTimeout(1000,()=>{s.destroy();resolve(false)});
 })));
 const udp=await new Promise(resolve=>{
   const s=dgram.createSocket('udp4');
   s.on('error',e=>{s.close();resolve(['EPERM','EACCES'].includes(e.code))});
   s.send('synthetic-boundary-probe',443,'192.0.2.1',e=>{
     s.close();resolve(Boolean(e)&&['EPERM','EACCES'].includes(e.code));
   });
 });
 let account=false;
 try{fs.readFileSync(process.argv[2])}
 catch(e){account=['EPERM','EACCES'].includes(e.code)};
 if(!denied.every(Boolean)||!udp||!account)throw new Error('boundary_failed');
 console.log(JSON.stringify({tcp_udp_denied:3,account_file_denied:true,inherited_environment_absent:true}));
})().catch(()=>{process.stderr.write('node_boundary_refused\n');process.exitCode=1});
"""
        node = subprocess.run(
            process.wrap(runtime.node, ["-e", node_probe, str(port), str(fake_account)]),
            cwd=directory,
            env=process.environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        (directory / "node-boundary.log").write_text(node.stdout + node.stderr)
        if node.returncode:
            raise AssertionError("actual_node_boundary_probe_failed")
        node_result = json.loads(node.stdout)
    finally:
        listener.close()
    (directory / "provider.py").write_text("from servegauge.comparison.provider import call_api\n")
    (directory / "assertion.py").write_text("""
import json
def get_assert(output, context):
    value=json.loads(output) if isinstance(output,str) else output
    message=value['response']['choices'][0]['message']
    expected=context['vars']['expected_content']
    passed=message.get('content')==expected
    if expected=='é':
        events=value['sse_events']
        passed=passed and any(e['is_comment'] for e in events)
        passed=passed and any('"bytes":[195,169]' in e['data'] for e in events)
        passed=passed and len(message['tool_calls'])==1
    return {'pass':bool(passed),'score':1 if passed else 0,
            'reason':'Independent synthetic channel equality; not an owner failure diagnosis.'}
""")
    body = {
        "model": "synthetic-source",
        "stream": False,
        "messages": [{"role": "user", "content": "same exact original request"}],
    }
    rows, tests = [], []
    for index, content in enumerate(["first", "second", "é", None]):
        request = {**body, "stream": index == 2}
        attempt = Attempt(
            n=1,
            status=200,
            response={"choices": [{"message": {"content": content}}]},
            timings=Timings(start_unix=0, total_ms=0),
        )
        if index == 2:
            frames = (
                ": keep alive\n\nid: source\nevent: chunk\n"
                'data: {"choices":[{"index":0,"delta":{"content":"é",'
                '"tool_calls":[{"index":0,"id":"c","function":{"name":"convert",'
                '"arguments":"{}"}}]},"logprobs":{"content":[{"token":"é",'
                '"bytes":[195,169],"logprob":-0.1}]}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode()
            attempt.sse_events = SseParser().feed(frames, 0)
            attempt.response = assemble_stream(attempt.sse_events)[0]
            attempt.response_headers = {"content-type": "text/event-stream"}
        elif index == 3:
            attempt.status, attempt.response, attempt.error = None, None, "transport_timeout"
        record = Exchange(
            exchange_id=f"exchange-{index}",
            run_id="synthetic-source",
            endpoint_id="source",
            suite_id="S1",
            item_id=f"item-{index}",
            sample_idx=0,
            request=request,
            attempts=[attempt],
        )
        row = ReplayObservation(
            case_id="synthetic",
            role="primary",
            source_position=index,
            record_hash=content_hash(record.model_dump(mode="json")),
            request_hash=content_hash(request),
            record=record,
        )
        address = ReplayAddress(
            case_id="synthetic",
            role="primary",
            source_run_id=record.run_id,
            endpoint_id=record.endpoint_id,
            suite_id=record.suite_id,
            item_id=record.item_id,
            sample_idx=0,
            exchange_id=record.exchange_id,
            attempt_n=1,
        )
        rows.append(row.model_dump(mode="json"))
        tests.append(
            {
                "description": f"synthetic-{index}",
                "vars": {
                    "wire": canonical_json(request).decode(),
                    "source_address": address.model_dump(mode="json"),
                    "expected_content": "é" if index == 2 else "first",
                },
                "assert": [{"type": "python", "value": "file://assertion.py"}],
            }
        )
    tests.append({**tests[0], "description": "exhausted-source"})
    blob = canonical_json(rows)
    (directory / "replay-book.json").write_bytes(blob)
    config = {
        "providers": [
            {
                "id": "file://provider.py",
                "config": {
                    "book": "replay-book.json",
                    "book_sha256": hashlib.sha256(blob).hexdigest(),
                },
            }
        ],
        "prompts": ["{{wire}}"],
        "tests": tests,
        "sharing": False,
        "evaluateOptions": {"maxConcurrency": 1, "cache": False, "showProgressBar": False},
    }
    (directory / "promptfoo.json").write_bytes(canonical_json(config))
    source_hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in [
            "provider.py",
            "assertion.py",
            "replay-book.json",
            "promptfoo.json",
            "boundary.sb",
            "empty.env",
        ]
    }
    result = subprocess.run(
        process.eval_command(),
        cwd=directory,
        env=process.environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (directory / "promptfoo.log").write_text(result.stdout + result.stderr)
    if not (directory / "promptfoo-results.json").is_file():
        raise AssertionError("actual_external_tool_did_not_produce_results")
    external = json.loads((directory / "promptfoo-results.json").read_text())
    (directory / "observed-shape.json").write_text(
        json.dumps({"exit": result.returncode, "top_keys": list(external)}, indent=2)
    )
    summary = external["results"]
    records = summary["results"]
    assert (
        len(records) == 5
        and summary["stats"]["successes"] == 2
        and summary["stats"]["failures"] == 1
        and summary["stats"]["errors"] == 2
    )
    assert result.returncode == 100
    audit = json.loads((directory / "provider-audit.json").read_text())
    assert audit["tape"]["consumed_attempts"] == 4 and audit["tape"]["remaining_slots"] == 0
    assert audit["harness_refused"] and audit["tape"]["refusals"]
    assert audit["pending_reservations"] == audit["new_spend_usd"] == 0
    assert all(
        hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name, digest in source_hashes.items()
    )
    ledger = Ledger(directory)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    return {
        "synthetic_only": True,
        "owner_comparison_executed": False,
        "tool_version": "0.124.1",
        "actual_cli_exit": result.returncode,
        "test_cases": 5,
        "passing_controls": 2,
        "failing_controls": 1,
        "errors": 2,
        "source_attempt_slots": 4,
        "consumed_attempts": 4,
        "cache_did_not_reuse_identical_prompts": True,
        "captured_sse_comments_token_bytes_and_tool_fields_retained": True,
        "source_timeout_and_exhaustion_remain_errors": True,
        "python_boundary": python_result,
        "node_boundary": node_result,
        "input_sha256": source_hashes,
        "pending_reservations": 0,
        "new_spend_usd": 0,
    }
