"""Daytona REST helpers for the SkyPilot provisioner.

Plain ``requests`` against ``https://app.daytona.io/api`` -- deliberately no
vendor SDK, so ``pip install skypilot[daytona]`` needs nothing beyond what
SkyPilot already ships. Daytona's whole surface is a small REST API and the
only credential is ``DAYTONA_API_KEY``.

Things learned from the API that shape this module
--------------------------------------------------
* ``GET /organizations`` answers **401 Invalid credentials** to a perfectly
  valid sandbox-scoped API key, and ``.../usage`` answers 403 without billing
  scope. A liveness check written against ``/organizations`` therefore reports
  every working key as dead. :func:`whoami` probes ``/api-keys/current``.
* ``POST /sandbox`` has **no ``image`` field**, whatever the docs' curl example
  shows. Sending one is silently ignored, the platform falls back to the
  default snapshot, and the request is then rejected with
  ``400 "Cannot specify Sandbox resources when using a snapshot"`` -- a 400
  that names neither the real cause nor the offending field. An image travels
  as ``buildInfo.dockerfileContent``.
* A **GPU sandbox is ephemeral by construction**: it must be created with
  ``autoDeleteInterval: 0`` (deleted the moment it stops), its filesystem does
  not survive a stop, and auto-pause is unsupported. There is no "stopped"
  state to come back to, which is why ``STOP`` is an unsupported feature on
  this cloud.
* ``ttlMinutes`` is a wall-clock deadline Daytona enforces **server-side**,
  regardless of sandbox state. It is the teardown that survives the API server
  dying, and it is always set.
* Spot GPU creates **fail immediately** when there is no capacity rather than
  queueing, and a running spot sandbox can be destroyed with no notice and no
  webhook. ``spotEvictedAt`` on the corpse (readable for 24h) is the only way
  to tell preemption from a crash.
"""

import json
import os
import time
import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.adaptors import common as adaptors_common
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    import requests
else:
    requests = adaptors_common.LazyImport('requests')

logger = sky_logging.init_logger(__name__)

DEFAULT_API_URL = 'https://app.daytona.io/api'
#: Daytona's SSH gateway. Auth is the access token as the *username*; see
#: :func:`create_ssh_access`.
SSH_GATEWAY_HOST = 'ssh.app.daytona.io'
SSH_GATEWAY_PORT = 22

CREDENTIAL_FILE = '~/.daytona/config.json'
API_KEY_ENV_VAR = 'DAYTONA_API_KEY'
API_URL_ENV_VAR = 'DAYTONA_API_URL'
ORG_ID_ENV_VAR = 'DAYTONA_ORGANIZATION_ID'

#: The label that maps a Daytona sandbox back to a SkyPilot cluster. Lookup is
#: by label and never by ``name``: a sandbox name is unique per organization,
#: so naming the box after the cluster would make a relaunch collide with a
#: not-yet-reaped predecessor.
CLUSTER_LABEL = 'sky-cluster-name'
#: Marks a sandbox as SkyPilot's. Keeps teardown and orphan sweeps away from
#: the agent sandboxes sharing the same Daytona account.
OWNER_LABEL = 'sky-owner'
OWNER_VALUE = 'skypilot'

REQUEST_TIMEOUT_S = 60
#: States that mean "this sandbox is on its way out"; never reuse one.
DEAD_STATES = frozenset(
    {'destroyed', 'destroying', 'error', 'build_failed', 'archived'})
RUNNING_STATES = frozenset({'started'})


class DaytonaError(RuntimeError):
    """A Daytona API call failed."""


def _api_url() -> str:
    return os.environ.get(API_URL_ENV_VAR, DEFAULT_API_URL).rstrip('/')


def api_key() -> str:
    """``DAYTONA_API_KEY``, else ``~/.daytona/config.json``.

    Env first because that is what Daytona's own SDK and CLI read.
    """
    env = os.environ.get(API_KEY_ENV_VAR, '').strip()
    if env:
        return env
    path = os.path.expanduser(CREDENTIAL_FILE)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DaytonaError(
            f'No Daytona credential: set {API_KEY_ENV_VAR} or write '
            f'{CREDENTIAL_FILE}. ({e})') from e
    key = (data or {}).get('api_key')
    if not key:
        raise DaytonaError(f'{CREDENTIAL_FILE} has no "api_key".')
    return str(key)


def _request(method: str,
             path: str,
             *,
             params: Optional[Dict[str, Any]] = None,
             body: Optional[Dict[str, Any]] = None,
             timeout: int = REQUEST_TIMEOUT_S) -> Any:
    url = f'{_api_url()}/{path.lstrip("/")}'
    headers = {
        'Authorization': f'Bearer {api_key()}',
        'Content-Type': 'application/json',
        'User-Agent': 'skypilot-daytona',
    }
    try:
        resp = requests.request(method,
                                url,
                                headers=headers,
                                params=params,
                                json=body,
                                timeout=timeout)
    except Exception as e:  # pylint: disable=broad-except
        raise DaytonaError(f'Daytona API unreachable ({method} {path}): '
                           f'{e}') from e
    if resp.status_code >= 400:
        detail = resp.text[:400]
        raise DaytonaError(f'Daytona API {method} {path} failed with HTTP '
                           f'{resp.status_code}: {detail}')
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return resp.text


def whoami() -> Dict[str, Any]:
    """The current API key's record -- the cheapest authenticated call.

    Not ``/organizations``: that endpoint 401s for a valid sandbox-scoped API
    key, so using it as a liveness probe reports working keys as dead.
    """
    got = _request('GET', 'api-keys/current')
    return got if isinstance(got, dict) else {}


def organization_id() -> Optional[str]:
    """The org id, read off any org-scoped record.

    ``/api-keys/current`` does not carry it and ``GET /organizations`` is
    closed to API keys, so it comes from a snapshot (preferred: an account with
    no sandboxes still has snapshots) or a sandbox.
    """
    env = os.environ.get(ORG_ID_ENV_VAR, '').strip()
    if env:
        return env
    for path in ('snapshots', 'sandbox'):
        try:
            got = _request('GET', path, params={'limit': 1})
        except DaytonaError:
            continue
        items = got.get('items') if isinstance(got, dict) else got
        for item in (items or []):
            org = (item or {}).get('organizationId')
            if org:
                return str(org)
    return None


def gpu_regions() -> List[str]:
    """Regions whose GPU sandbox class is offered to this org right now.

    The only pre-create capacity signal Daytona has, and it is coarse: it says
    the org can get *a* GPU in a region, never which type or how many. Empty is
    a real "no"; non-empty is not a promise -- a spot create can still fail
    immediately for want of capacity.
    """
    org = organization_id()
    if org is None:
        return []
    try:
        rows = _request('GET', f'organizations/{org}/available-sandbox-classes')
    except DaytonaError:
        return []
    out = []
    for row in (rows or []):
        if isinstance(row, dict) and row.get('gpuAvailable') and row.get(
                'regionId'):
            out.append(str(row['regionId']))
    return sorted(set(out))


#: Packages SkyPilot's setup commands assume exist and a container image
#: usually does not. Measured on a real launch against
#: ``pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime``: SETUP died at
#: ``bash: curl: command not found`` (conda and uv are both installed by
#: curl-piped scripts) and at ``E: Unable to locate package patch``, because
#: the image ships with empty apt lists.
#:
#: Installing them at *build* time rather than at boot is deliberate. Daytona
#: builds the image on its own side, where egress is unrestricted; a sandbox's
#: own egress can be restricted to an allow-list on lower org tiers, so an
#: ``apt-get update`` from inside the box is exactly the thing that fails on a
#: locked account. Building also means the layer is cached and the cost is paid
#: once per image, not once per launch.
SETUP_PACKAGES = ('curl', 'patch', 'rsync', 'procps')


def dockerfile_for(image: str) -> str:
    """The one-image build context Daytona takes in ``buildInfo``.

    ``|| true`` keeps a non-Debian base usable rather than failing the build;
    :func:`bootstrap_node` is what turns a genuinely unusable image into a
    precise error, after boot, naming the image and the missing tool.
    """
    packages = ' '.join(SETUP_PACKAGES)
    return (f'FROM {image}\n'
            f'RUN (command -v curl >/dev/null && command -v patch >/dev/null) '
            f'|| (apt-get update && apt-get install -y --no-install-recommends '
            f'{packages} && rm -rf /var/lib/apt/lists/*) || true\n')


def organization_usage() -> List[Dict[str, Any]]:
    """Per-region quota and current usage. **Costs nothing.**

    ``GET /organizations/{id}/usage`` is the only free pre-create signal
    Daytona has, and it answers a different question from
    :func:`gpu_regions`: not "is there stock" but "does this org have room".
    It returns, per region and sandbox class, ``totalGpuQuota`` /
    ``currentGpuUsage`` (plus the same for cpu/memory/disk) and the per-GPU
    ceilings ``maxCpuPerGpu`` / ``maxMemoryPerGpu`` / ``maxDiskPerGpu``.

    Note the key scope: a plain ``write:sandboxes`` API key gets **403** here.
    Callers must treat "cannot read usage" as "unknown", never as "no room".
    """
    org = organization_id()
    if org is None:
        return []
    try:
        got = _request('GET', f'organizations/{org}/usage')
    except DaytonaError:
        return []
    rows = got.get('regionUsage') if isinstance(got, dict) else got
    return [r for r in (rows or []) if isinstance(r, dict)]


def check_gpu_quota(region: str, gpu_count: int) -> Optional[str]:
    """Why this GPU ask cannot fit the org's quota, or None if it can.

    Free, and it converts one whole class of launch failure from a 400 during
    provisioning into a refusal before anything is created. It says nothing
    about *stock*: Daytona exposes no per-GPU-type availability endpoint, so
    whether an H100 spot runner is free right now is only discoverable by
    asking for one (a create that fails immediately with "No available GPU
    spot runners"). That is why the SkyPilot catalog for this cloud reports
    price without claiming capacity.
    """
    if gpu_count <= 0:
        return None
    for row in organization_usage():
        if row.get('regionId') != region or row.get('sandboxClass') != 'container':
            continue
        total = row.get('totalGpuQuota')
        used = row.get('currentGpuUsage') or 0
        if total is None:
            return None
        if used + gpu_count > total:
            return (f'Daytona GPU quota in {region} is {total} and '
                    f'{used} are already in use, so a {gpu_count}-GPU sandbox '
                    'does not fit. Free some sandboxes or raise the quota.')
        return None
    return None


#: Domains a SkyPilot node must reach to build itself.
#:
#: Daytona restricts a sandbox's **runtime** egress to a built-in "essential"
#: list on lower organization tiers; everything else is connection-reset with
#: no error message worth reading. Measured: with the default list, SETUP dies
#: at ``curl: (35) Recv failure: Connection reset by peer`` fetching the uv
#: installer from ``astral.sh``.
#:
#: Note the asymmetry that makes this confusing to debug: the **image build**
#: runs on Daytona's side with unrestricted egress, so ``apt-get install`` in
#: the build layer succeeds while the identical command inside the running
#: sandbox fails.
#:
#: Setting ``domainAllowList`` **REPLACES** the tier default rather than adding
#: to it, so this list has to name everything -- uv, conda, pypi, github, the
#: torch index and the HF CDN. A Tier 3/4 organization has full internet and
#: needs none of it; ``domain_allow_list: []`` in ``~/.sky/config.yaml`` under
#: ``daytona`` turns it off.
#: Daytona caps a domain allow list at **20 entries** (measured: 24 -> HTTP 400
#: "Domain allow list cannot contain more than 20 domains").
MAX_DOMAIN_ALLOW_LIST = 20

#: Daytona's "essential services" are reachable on every tier with no
#: configuration, and they cover almost everything a SkyPilot + SkyRL node
#: needs: pypi, ``astral.sh`` (uv), ``repo.anaconda.com``, github,
#: ``pytorch.org``, huggingface, ubuntu/debian.
#: (https://www.daytona.io/docs/en/network-limits.md#essential-services)
#:
#: **Almost.** SkyRL's lockfile pulls vLLM from ``wheels.vllm.ai``, which is
#: not an essential service, so on a Tier 1/2 organization it is
#: connection-reset: ``Failed to download vllm==0.23.0+cu129 ... Connection
#: reset by peer (os error 104)``. Everything else in the sync succeeds --
#: Megatron even builds from github -- and the install dies on that one host.
#:
#: There is no way to *add* a domain: ``domainAllowList`` REPLACES the tier
#: default. So this list has to restate the essential set we depend on and add
#: the missing host, and it must stay within Daytona's cap of 20 entries
#: (measured: 24 -> HTTP 400). Wildcards keep it to 18 with headroom.
#:
#: Setting it also makes Daytona inject ``HTTP_PROXY``/``HTTPS_PROXY``, and
#: that proxy is HTTP/1.1 CONNECT only -- which is what broke ``uv sync`` with
#: ``tunnel error: unsuccessful`` the first time round. :func:`bootstrap_node`
#: is where that is neutralised, because a ``no_proxy`` set at create time is
#: overwritten by Daytona's own runtime injection.
MAX_DOMAIN_ALLOW_LIST = 20

DEFAULT_DOMAIN_ALLOW_LIST: tuple = (
    # The one host that is NOT an essential service and that SkyRL cannot do
    # without. This entry is the whole reason the list exists.
    'wheels.vllm.ai',
    # PyPI
    'pypi.org',
    '*.pythonhosted.org',
    'bootstrap.pypa.io',
    # uv
    'astral.sh',
    '*.astral.sh',
    # conda
    'repo.anaconda.com',
    # GitHub (SkyRL clone; Megatron-LM and Megatron-Bridge build from source)
    'github.com',
    '*.github.com',
    '*.githubusercontent.com',
    # Torch wheels
    '*.pytorch.org',
    # Hugging Face model downloads
    'huggingface.co',
    '*.huggingface.co',
    'hf.co',
    '*.hf.co',
    '*.xethub.hf.co',
    # Distro archives
    '*.ubuntu.com',
    '*.debian.org',
)


def shell_quote(value: str) -> str:
    """Single-quote a value for a POSIX shell."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def no_proxy_value(allow: Optional[List[str]] = None) -> str:
    """The ``no_proxy`` a sandbox needs when an allow list is in force.

    Two distinct things have to bypass the injected CONNECT proxy:

    * **the allow-listed hosts**, because the proxy is HTTP/1.1 only and uv
      negotiates HTTP/2 (``tunnel error: unsuccessful``), and
    * **loopback and the sandbox's own private address**, because ray talks to
      itself over gRPC and a proxied local connection is how GCS ends up
      unreachable from the node hosting it.

    This does not widen egress. With a ``domainAllowList`` in force the
    firewall still decides what leaves the box; ``no_proxy`` only stops
    well-behaved clients wrapping allowed traffic in CONNECT.
    """
    hosts = ['localhost', '127.0.0.1', '::1',
             '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
    for domain in (allow if allow is not None else DEFAULT_DOMAIN_ALLOW_LIST):
        hosts.append(str(domain).lstrip('*.'))
    seen, out = set(), []
    for h in hosts:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return ','.join(out)


# -- sandboxes --------------------------------------------------------------


def find_sandboxes(cluster_name_on_cloud: str,
                   include_dead: bool = False) -> List[Dict[str, Any]]:
    """Every sandbox belonging to a SkyPilot cluster."""
    labels = json.dumps({
        CLUSTER_LABEL: cluster_name_on_cloud,
        OWNER_LABEL: OWNER_VALUE,
    })
    got = _request('GET', 'sandbox', params={'labels': labels})
    items = got.get('items') if isinstance(got, dict) else got
    out = []
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        state = str(item.get('state') or '').lower()
        if not include_dead and state in DEAD_STATES:
            continue
        out.append(item)
    return out


def get_sandbox(sandbox_id: str) -> Optional[Dict[str, Any]]:
    try:
        got = _request('GET', f'sandbox/{sandbox_id}')
    except DaytonaError as e:
        if '404' in str(e):
            return None
        raise
    return got if isinstance(got, dict) else None


def create_sandbox(cluster_name_on_cloud: str, node_config: Dict[str,
                                                                 Any]) -> str:
    """Create one GPU (or CPU) sandbox for a cluster and return its id."""
    gpu_type = node_config.get('GpuType')
    gpu_count = int(node_config.get('Gpu') or 0)
    body: Dict[str, Any] = {
        'cpu': int(node_config['Cpu']),
        'memory': int(node_config['Memory']),
        'disk': int(node_config['Disk']),
        # A GPU sandbox is ephemeral by construction and Daytona rejects
        # anything else; a CPU one is held to the same rule so teardown has one
        # meaning on this cloud.
        'autoDeleteInterval': 0,
        # No idle reaping: a training poll goes quiet without being idle, and
        # SkyPilot's own autostop is unsupported here. ttlMinutes below is what
        # actually bounds the spend.
        'autoStopInterval': 0,
        'labels': {
            CLUSTER_LABEL: cluster_name_on_cloud,
            OWNER_LABEL: OWNER_VALUE,
        },
    }
    if gpu_count:
        body['gpu'] = gpu_count
        body['gpuType'] = [gpu_type]
        # GPU-only. Daytona rejects `spot` when the sandbox requests no GPUs.
        body['spot'] = bool(node_config.get('UseSpot'))
    ttl_minutes = node_config.get('TtlMinutes')
    if ttl_minutes:
        body['ttlMinutes'] = int(ttl_minutes)
    env = node_config.get('Env') or {}
    if env:
        body['env'] = dict(env)
    allow = node_config.get('DomainAllowList')
    if allow is None:
        allow = list(DEFAULT_DOMAIN_ALLOW_LIST)
    if allow:
        if len(allow) > MAX_DOMAIN_ALLOW_LIST:
            raise DaytonaError(
                f'Daytona accepts at most {MAX_DOMAIN_ALLOW_LIST} domains in '
                f'domainAllowList; {len(allow)} were given. Collapse them with '
                'wildcards (e.g. "*.huggingface.co").')
        # Setting a domainAllowList makes Daytona inject HTTP(S)_PROXY into the
        # sandbox, and that proxy is HTTP/1.1 CONNECT only -- it cannot carry
        # HTTP/2 or gRPC. Measured: `uv sync` dies with
        # `tunnel error: unsuccessful` fetching torch, because uv negotiates
        # HTTP/2. Naming the allow-listed domains in `no_proxy` makes clients
        # talk to them directly instead of through the tunnel.
        #
        # This does NOT widen egress: with a domainAllowList, web-port traffic
        # is redirected at the network layer, which a sandbox cannot bypass by
        # clearing an environment variable. It only stops well-behaved clients
        # from wrapping their own traffic in CONNECT.
        # NOT set as create-time `env`: Daytona injects its own
        # `no_proxy=localhost,127.0.0.1,::1` at runtime and overwrites it.
        # bootstrap_node writes it to /etc/environment and ~/.bashrc instead,
        # which is what SETUP's `bash --login` actually reads.
        # Only set when non-empty: an empty value would REPLACE the tier
        # default with nothing and cut off ALL egress.
        body['domainAllowList'] = ','.join(allow)
    region = node_config.get('Region')
    if region:
        # Recorded, not steering: Daytona ignores the target for GPU sandboxes
        # because of scarcity. A GPU lands where there is capacity.
        body['target'] = region
    snapshot = node_config.get('Snapshot')
    if snapshot:
        # A snapshot carries its own shape and the API rejects the combination
        # ("Cannot specify Sandbox resources when using a snapshot").
        for key in ('cpu', 'memory', 'disk', 'gpu', 'gpuType'):
            body.pop(key, None)
        body['snapshot'] = snapshot
    else:
        image = node_config.get('DockerImage') or node_config['DefaultImage']
        # NOT an `image` field -- see the module docstring.
        body['buildInfo'] = {'dockerfileContent': dockerfile_for(image)}
    public_key = node_config.get('PublicKey')
    if public_key:
        # Belt and braces beside the gateway token: if the image runs its own
        # sshd, SkyPilot's cluster key still gets in.
        body.setdefault('env', {})['SKY_PUBLIC_KEY'] = public_key
    got = _request('POST', 'sandbox', body=body)
    if not isinstance(got, dict) or not got.get('id'):
        raise DaytonaError(f'Unexpected Daytona create response: {got!r}')
    return str(got['id'])


def wait_started(sandbox_id: str, timeout_s: int) -> Dict[str, Any]:
    """Block until the sandbox is started, or say precisely why it is not."""
    deadline = time.time() + timeout_s
    last_state = '?'
    while True:
        record = get_sandbox(sandbox_id)
        if record is None:
            raise DaytonaError(
                f'Daytona sandbox {sandbox_id} vanished before it started.')
        last_state = str(record.get('state') or '').lower()
        if last_state in RUNNING_STATES:
            return record
        if last_state in DEAD_STATES:
            evicted = record.get('spotEvictedAt')
            reason = record.get('errorReason')
            if evicted:
                raise DaytonaError(
                    f'Daytona sandbox {sandbox_id} was PREEMPTED at {evicted} '
                    'before it started -- spot capacity was reclaimed for an '
                    'on-demand sandbox, without notice.')
            raise DaytonaError(
                f'Daytona sandbox {sandbox_id} entered state {last_state!r} '
                f'before starting: {reason or "no reason given"}')
        if time.time() >= deadline:
            raise DaytonaError(
                f'Daytona sandbox {sandbox_id} was still {last_state!r} after '
                f'{timeout_s}s.')
        time.sleep(5)


def delete_sandbox(sandbox_id: str) -> None:
    """Delete a sandbox. Idempotent; a missing sandbox is a success."""
    for params in ({}, {'force': 'true'}):
        try:
            _request('DELETE', f'sandbox/{sandbox_id}', params=params or None)
            return
        except DaytonaError as e:
            if '404' in str(e):
                return
            if params:
                raise
            logger.debug(f'Daytona delete of {sandbox_id} needs force: {e}')


def wait_gone(sandbox_id: str, timeout_s: int = 180) -> Optional[str]:
    """Poll until the sandbox is really gone. Returns its state if it is not.

    Nothing else reaps a leaked Daytona GPU except its ``ttlMinutes``
    deadline, so teardown asserts its post-condition rather than assuming the
    DELETE landed.
    """
    deadline = time.time() + timeout_s
    while True:
        record = get_sandbox(sandbox_id)
        if record is None:
            return None
        state = str(record.get('state') or '').lower()
        if state in ('destroyed',):
            return None
        if time.time() >= deadline:
            return state
        time.sleep(5)


def preempted_at(sandbox_id: str) -> Optional[str]:
    """``spotEvictedAt``, or None. Readable for 24h after preemption."""
    record = get_sandbox(sandbox_id)
    if record is None:
        return None
    return record.get('spotEvictedAt') or None


def get_snapshot(name_or_id: str) -> Optional[Dict[str, Any]]:
    """A snapshot record, or None when it does not exist."""
    try:
        got = _request('GET', f'snapshots/{name_or_id}')
    except DaytonaError as e:
        if '404' in str(e):
            return None
        raise
    return got if isinstance(got, dict) else None


def check_snapshot_shape(name: str,
                         node_config: Dict[str, Any]) -> Optional[str]:
    """Why booting this snapshot would not give the planned machine, or None.

    A Daytona snapshot fixes the machine shape as well as the filesystem:
    ``POST /api/sandbox`` takes ``snapshot`` **instead of**
    cpu/memory/disk/gpu and rejects a create that sends both. So a snapshot
    whose baked shape differs from the one SkyPilot chose would silently hand
    back a different machine than was planned -- and than was **priced**,
    which is the part that matters. Better to say so than to bill a surprise.

    Returns None when the snapshot is missing: that is a separate failure with
    its own message at create time, and guessing here would hide it.
    """
    record = get_snapshot(name)
    if record is None:
        return (f'Daytona snapshot {name!r} does not exist. Build it with '
                'scripts/publish_daytona_snapshot.py in evsys-enterprise, or '
                'clear the `daytona.snapshot` config to take the slower '
                'image path.')
    if str(record.get('state') or '').lower() != 'active':
        return (f'Daytona snapshot {name!r} is in state '
                f'{record.get("state")!r}, not active'
                + (f': {record["errorReason"]}' if record.get('errorReason')
                   else '.'))
    mismatches = []
    for field, key in (('Cpu', 'cpu'), ('Memory', 'memory'),
                       ('Disk', 'disk'), ('Gpu', 'gpu')):
        want = node_config.get(field)
        got = record.get(key)
        if want is None or got is None:
            continue
        if int(want) != int(got):
            mismatches.append(f'{key}: snapshot has {got}, plan wants {want}')
    want_type = node_config.get('GpuType')
    got_types = [str(t) for t in (record.get('gpuType') or [])]
    if want_type and got_types and want_type not in got_types:
        mismatches.append(
            f'gpuType: snapshot has {"/".join(got_types)}, plan wants '
            f'{want_type}')
    if mismatches:
        return (f'Daytona snapshot {name!r} was baked for a different machine '
                f'than SkyPilot planned ({"; ".join(mismatches)}). A snapshot '
                'carries its resources, so booting it would quietly give you '
                'the baked shape at the planned price. Bake a snapshot for '
                'this shape or pin the matching resources.')
    return None


# -- reachability -----------------------------------------------------------


def create_ssh_access(sandbox_id: str, expires_in_minutes: int) -> str:
    """Mint an SSH access token for a sandbox.

    Daytona's SSH gateway authenticates by putting the **token in the username
    position** -- ``ssh <token>@ssh.app.daytona.io`` -- rather than by key. The
    key SkyPilot passes with ``-i`` is therefore ignored by the gateway, which
    is why the cluster's ``ssh_user`` is this token and not a login name.

    Tokens expire, so this is re-minted on every ``get_cluster_info()`` rather
    than once at provision: a cluster that outlives its token would otherwise
    become unreachable for the rest of its life.
    """
    got = _request('POST',
                   f'sandbox/{sandbox_id}/ssh-access',
                   params={'expiresInMinutes': int(expires_in_minutes)})
    token = (got or {}).get('token') if isinstance(got, dict) else None
    if not token:
        raise DaytonaError(
            f'Daytona did not return an SSH access token for {sandbox_id}: '
            f'{got!r}')
    return str(token)


def signed_preview_url(sandbox_id: str, port: int,
                       expires_in_seconds: int) -> str:
    """A preview URL with the token **embedded in the URL**, not a header.

    This is what makes a served port usable by an ordinary HTTP client.
    :func:`preview_url` returns a URL plus a token that must travel in an
    ``x-daytona-preview-token`` header, and most clients cannot be told to add
    one -- the ``tinker`` client that talks to a SkyRL server certainly cannot,
    it just takes a base URL. A signed URL
    (``https://{port}-{token}.{proxyDomain}``) needs no headers at all, which
    is why SkyPilot's endpoint for this cloud is the signed one.

    The two token kinds are **not interchangeable**: a standard preview token
    cannot be used in a signed URL and vice versa.

    Security note, because this trades one property for another: anyone
    holding the URL can reach the port until the token expires, and SkyRL
    performs no authentication of its own. Hence an explicit, bounded expiry
    -- Daytona's default is 60 seconds, deliberately short, and the docs say
    to always set this rather than inherit it.
    """
    got = _request(
        'GET', f'sandbox/{sandbox_id}/ports/{int(port)}/signed-preview-url',
        params={'expiresInSeconds': int(expires_in_seconds)})
    if not isinstance(got, dict) or not got.get('url'):
        raise DaytonaError(
            f'No signed preview URL for {sandbox_id} port {port}: {got!r}')
    return str(got['url'])


def preview_url(sandbox_id: str, port: int) -> Tuple[str, str]:
    """``(url, token)`` for a port served inside the sandbox.

    Daytona has no public per-node IP. A served port is reached through an
    authenticated proxy at ``https://{port}-{sandboxId}.{proxyDomain}`` with an
    ``x-daytona-preview-token`` header, unless the sandbox was created
    ``public: true``.
    """
    got = _request('GET', f'sandbox/{sandbox_id}/ports/{int(port)}/preview-url')
    if not isinstance(got, dict) or not got.get('url'):
        raise DaytonaError(
            f'No Daytona preview URL for {sandbox_id} port {port}: {got!r}')
    return str(got['url']), str(got.get('token') or '')


def check_credentials() -> Tuple[bool, Optional[str]]:
    """Whether this key exists, works, and may create sandboxes."""
    try:
        who = whoami()
    except DaytonaError as e:
        with ux_utils.print_exception_no_traceback():
            return False, (
                f'Daytona rejected the credential. Set {API_KEY_ENV_VAR} (or '
                f'write {CREDENTIAL_FILE}) with a key that has '
                f'write:sandboxes. ({e})')
    permissions = [str(p) for p in (who.get('permissions') or [])]
    if 'write:sandboxes' not in permissions:
        return False, ('The Daytona API key is valid but cannot create '
                       f'sandboxes (permissions: '
                       f'{", ".join(permissions) or "none"}).')
    return True, None


# -- the HTTPS command + file transport -------------------------------------
#
# This is what makes Daytona rentable from inside a locked agent sandbox.
# Every call below is HTTPS to app.daytona.io / proxy.app.daytona.io; nothing
# needs port 22. SkyPilot provisions Verda over SSH, and a sandbox egress proxy
# blocks port 22, which is why Modal was until now the only cloud such an agent
# could rent from. Daytona joins it on the same terms.

TOOLBOX_URL = 'https://proxy.app.daytona.io/toolbox'
#: Default timeout for one exec. SkyPilot's setup steps are long.
DEFAULT_EXEC_TIMEOUT_S = 1800


def _toolbox_request(method: str,
                     path: str,
                     *,
                     params: Optional[Dict[str, Any]] = None,
                     body: Optional[Dict[str, Any]] = None,
                     files: Optional[Dict[str, Any]] = None,
                     timeout: int = REQUEST_TIMEOUT_S,
                     raw: bool = False) -> Any:
    url = f'{TOOLBOX_URL}/{path.lstrip("/")}'
    headers = {
        'Authorization': f'Bearer {api_key()}',
        'User-Agent': 'skypilot-daytona',
    }
    if body is not None:
        headers['Content-Type'] = 'application/json'
    try:
        resp = requests.request(method,
                                url,
                                headers=headers,
                                params=params,
                                json=body,
                                files=files,
                                timeout=timeout)
    except Exception as e:  # pylint: disable=broad-except
        raise DaytonaError(f'Daytona toolbox unreachable ({method} {path}): '
                           f'{e}') from e
    if resp.status_code >= 400:
        raise DaytonaError(f'Daytona toolbox {method} {path} failed with HTTP '
                           f'{resp.status_code}: {resp.text[:400]}')
    if raw:
        return resp.content
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return resp.text


def exec_command(sandbox_id: str,
                 command: str,
                 *,
                 cwd: Optional[str] = None,
                 envs: Optional[Dict[str, str]] = None,
                 timeout_s: Optional[int] = None) -> Tuple[int, str]:
    """Run a shell command in the sandbox -> ``(exit_code, merged_output)``.

    The output is **merged**: the toolbox folds the sandbox's stderr into the
    same field as its stdout, with no flag to separate them. Measured, not
    assumed. Everything else measured well: a 1 MB command and 1 MB of output
    both survived intact, and exit codes propagate -- so unlike the Modal exec
    channel there is no chunking, no staging and no truncation recovery here.
    """
    timeout = int(timeout_s or DEFAULT_EXEC_TIMEOUT_S)
    body: Dict[str, Any] = {'command': command, 'timeout': timeout}
    if cwd:
        body['cwd'] = cwd
    if envs:
        body['envs'] = dict(envs)
    got = _toolbox_request('POST',
                           f'{sandbox_id}/process/execute',
                           body=body,
                           timeout=timeout + 60)
    if not isinstance(got, dict):
        return 0, str(got)
    return int(got.get('exitCode') or 0), str(got.get('result') or '')


def upload_file(sandbox_id: str, local_path: str, remote_path: str) -> None:
    """Upload one local file to an explicit remote path.

    The destination is passed to the API verbatim. Nothing here derives a
    destination from the source's basename, which is the failure mode that put
    a file under its local temp name instead of at its target in the Modal
    transfer path (``ev-sys/skypilot#4``).
    """
    with open(local_path, 'rb') as f:
        _toolbox_request('POST',
                         f'{sandbox_id}/files/upload',
                         params={'path': remote_path},
                         files={'file': (os.path.basename(remote_path), f)},
                         timeout=600)


def download_file(sandbox_id: str, remote_path: str) -> bytes:
    """Download one remote file. Binary-safe (verified, sha256 round trip)."""
    return _toolbox_request('GET',
                            f'{sandbox_id}/files/download',
                            params={'path': remote_path},
                            timeout=600,
                            raw=True)


def abs_remote_path(path: str) -> str:
    """Absolutise a remote path.

    Every remote path this provisioner emits is ``shlex.quote``d, so a leading
    ``~`` would never reach a shell to expand -- it would create a literal
    directory named ``~``.
    """
    if path.startswith('~'):
        return REMOTE_HOME + path[1:]
    if not path.startswith('/'):
        return f'{REMOTE_HOME}/{path}'
    return path


#: Daytona's stock images run as root, so ``$HOME`` is ``/root``.
REMOTE_HOME = '/root'

#: SkyPilot's setup commands are full of ``sudo``, and Daytona's stock images
#: (ubuntu:22.04, pytorch/pytorch:*) ship without it -- measured: ``which sudo``
#: is empty, ``whoami`` is ``root``. Installing it would mean an apt round trip
#: on every cold launch; since the sandbox is *already* root, a three-line shim
#: is equivalent and instant. It is deliberately only installed when the real
#: thing is absent, so an image that has proper sudo keeps it.
SUDO_SHIM = (
    'command -v sudo >/dev/null 2>&1 || { '
    'printf \'#!/bin/sh\\nexec "$@"\\n\' > /usr/local/bin/sudo && '
    'chmod +x /usr/local/bin/sudo; }')


def bootstrap_node(sandbox_id: str,
                   allow: Optional[List[str]] = None) -> None:
    """Make a fresh sandbox look enough like a VM for SkyPilot's setup.

    Two gaps, both measured on a live sandbox rather than guessed:

    * no ``sudo`` (see :data:`SUDO_SHIM`), and
    * no ``~/.ssh`` directory, which several of SkyPilot's setup steps append
      to before anything creates it.
    """
    no_proxy = no_proxy_value(allow)
    # Written into /etc/environment AND ~/.bashrc rather than passed as create
    # `env`: Daytona injects its own `no_proxy=localhost,127.0.0.1,::1` at
    # runtime, which OVERWRITES anything set at create time (measured). SETUP
    # runs under `bash --login`, so ~/.bashrc is the file that reaches it.
    proxy_fix = ''
    if no_proxy:
        proxy_fix = (
            f'grep -q SKY_NO_PROXY {REMOTE_HOME}/.bashrc 2>/dev/null || '
            f'printf \'# SKY_NO_PROXY\\nexport no_proxy=%s\\n'
            f'export NO_PROXY=%s\\n\' {shell_quote(no_proxy)} '
            f'{shell_quote(no_proxy)} >> {REMOTE_HOME}/.bashrc; '
            f'printf \'no_proxy=%s\\nNO_PROXY=%s\\n\' '
            f'{shell_quote(no_proxy)} {shell_quote(no_proxy)} '
            '>> /etc/environment; ')
    rc, out = exec_command(
        sandbox_id,
        f'set -e; {SUDO_SHIM}; mkdir -p {REMOTE_HOME}/.ssh; '
        f'chmod 700 {REMOTE_HOME}/.ssh; touch {REMOTE_HOME}/.ssh/config; '
        f'{proxy_fix}'
        'command -v sudo >/dev/null && echo SKY_BOOTSTRAP_OK',
        timeout_s=180)
    if rc != 0 or 'SKY_BOOTSTRAP_OK' not in out:
        raise DaytonaError(
            f'Could not bootstrap Daytona sandbox {sandbox_id} '
            f'(rc={rc}): {out[:400]}')
    # Fail here, with the tool named, rather than 10 minutes later inside
    # SETUP with `bash: curl: command not found`.
    missing = []
    for tool in ('curl', 'patch'):
        rc, _ = exec_command(sandbox_id, f'command -v {tool}', timeout_s=60)
        if rc != 0:
            missing.append(tool)
    if missing:
        raise DaytonaError(
            f'Daytona sandbox {sandbox_id} is missing {", ".join(missing)}, '
            "which SkyPilot's setup needs (conda and uv are installed by "
            'curl-piped scripts). The image build layer that installs them '
            'only works on a Debian/Ubuntu base -- pick such an image, or bake '
            'these tools into your own.')
