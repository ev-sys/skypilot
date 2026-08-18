"""Modal Server provisioning utilities.

This module backs a SkyPilot cluster with a Modal **Server** (``@app.server()``)
rather than a ``modal.Sandbox``. See ``instance.py`` for why.

Everything here is built on Modal's *public* API. That is a deliberate
constraint: an earlier design drove Modal's in-process exec internals
(``TaskCommandRouterClient`` / ``_ContainerProcess``) to get separated
stdout/stderr, but those are private and would break on a Modal upgrade with no
warning. Verified against modal 1.5.3.
"""

import base64
import functools
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.adaptors import modal as modal_adaptor

logger = sky_logging.init_logger(__name__)

TOKEN_ID_ENV_VAR = 'MODAL_TOKEN_ID'
TOKEN_SECRET_ENV_VAR = 'MODAL_TOKEN_SECRET'

# The Server object's name inside the per-cluster App. A cluster is one App
# holding one Server, so this is a constant.
SERVER_NAME = 'node'
_APP_PREFIX = 'skypilot'

# Modal's default Server port. The workload (SkyRL) binds this; we never do.
DEFAULT_PORT = 8000

# App tag keys. Modal App tags are how we recognise our own deployments and
# refuse to reuse one whose shape no longer matches the request.
TAG_CLUSTER = 'skypilot-cluster'
TAG_SPEC = 'skypilot-spec'
TAG_LIFETIME_DEADLINE = 'skypilot-max-lifetime-deadline'

# `modal container exec` truncates a command's output at 8192 characters when
# the process exits promptly: the CLI's `wait()` returns on the exit status
# while a separate task is still draining the output stream. Verified live --
# 256 KB round-trips perfectly once the remote command lingers, and is silently
# cut to exactly 8192 chars when it does not. Silent truncation of parsed
# output is a correctness bug, so every command this module runs ends with a
# drain delay AND a trailing sentinel that proves nothing was lost.
_DRAIN_TRUNCATION_LIMIT = 8192
_DRAIN_DELAY_SECONDS = 3

# `modal container exec` accepts roughly 8 KB of argv and then fails *silently*
# -- rc=0 with empty output, no error anywhere. Measured: 8 KB fine, 16 KB
# empty. We refuse to build a command near the cliff rather than let a
# successful-looking no-op through.
_ARGV_LIMIT_BYTES = 8 * 1024
_ARGV_SAFETY_MARGIN = 2 * 1024


def _modal():
    return modal_adaptor.modal


def app_name_for_cluster(cluster_name_on_cloud: str) -> str:
    """Modal App name holding this cluster's Server."""
    return f'{_APP_PREFIX}-{cluster_name_on_cloud}'


def get_modal_env_secret():
    """Return a Modal Secret carrying env-token credentials, if configured."""
    token_id = os.environ.get(TOKEN_ID_ENV_VAR)
    token_secret = os.environ.get(TOKEN_SECRET_ENV_VAR)
    if not token_id or not token_secret:
        return None
    return _modal().Secret.from_dict({
        TOKEN_ID_ENV_VAR: token_id,
        TOKEN_SECRET_ENV_VAR: token_secret,
    })


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------

_IMAGE_PYTHON_VERSION = '3.12'
# `procps`/`lsof` for SkyPilot's process probes, `rsync` because the *remote*
# side of a file sync still runs rsync locally inside the container, `curl` for
# readiness probes. No NVIDIA packages: Modal supplies the driver stack.
# `sudo` is required even though the container runs as root: SkyPilot's shared
# setup_commands invoke it unconditionally, and a missing binary fails the
# whole SETUP rather than being skipped.
_IMAGE_SYSTEM_PACKAGES = ('rsync', 'curl', 'procps', 'patch', 'lsof', 'tar',
                          'coreutils', 'sudo')


def build_image(named_image: Optional[str], docker_image: Optional[str],
                file_mounts: Optional[List[Dict[str, str]]]):
    """Build the Modal Image for a cluster node.

    ``named_image`` is a Modal **Named Image** such as
    ``evsys-train:42f3d44``. It cannot ride SkyPilot's ``image_id`` field --
    ``IMAGE_ID`` is an unsupported feature on this cloud -- so it is threaded
    through as its own node_config key and resolved here via
    ``Image.from_name()``. Note the single-argument ``name:tag`` form: there is
    no ``tag=`` keyword.

    ``file_mounts`` are baked into the image with ``add_local_file(copy=True)``
    rather than pushed after boot. This is the file-transfer transport for the
    provisioning path, and it is a deliberate choice: `modal container exec`
    forwards no stdin and caps argv near 8 KB, so it cannot carry a multi-MB
    SkyPilot wheel at any usable speed. Our transfer need is genuinely
    launch-time -- wheel, configs, credentials all land once -- which maps onto
    SkyPilot's own `workdir`/`file_mounts` semantics, so nothing is given up.
    """
    modal = _modal()
    if named_image is not None:
        if docker_image is not None:
            raise ValueError(
                'Modal node cannot request both a Modal Named Image '
                f'({named_image!r}) and a Docker image ({docker_image!r}).')
        # Single-arg 'name:tag'; Image.from_name() has no tag= kwarg.
        image = modal.Image.from_name(named_image)
    elif docker_image is not None:
        image = modal.Image.from_registry(docker_image,
                                          add_python=_IMAGE_PYTHON_VERSION)
    else:
        image = modal.Image.debian_slim(python_version=_IMAGE_PYTHON_VERSION)

    if named_image is None:
        # A Named Image is prebaked by us and already carries these; running
        # apt against it would defeat the point of the bake.
        image = image.apt_install(*_IMAGE_SYSTEM_PACKAGES)

    markers: Dict[str, str] = {}
    for mount in file_mounts or []:
        local_path = os.path.expanduser(mount['LocalPath'])
        remote_path = mount['RemotePath']
        if not os.path.exists(local_path):
            continue
        # Mirror rsync's placement exactly, because SkyPilot will later run the
        # equivalent rsync against these same pairs and must find them already
        # in place: a file lands AT the target, a directory lands INSIDE it.
        remote_path = abs_remote_path(remote_path)
        if os.path.isdir(local_path):
            dest = (f'{remote_path.rstrip("/")}/'
                    f'{os.path.basename(local_path.rstrip("/"))}')
            image = image.add_local_dir(local_path, dest, copy=True)
        else:
            dest = remote_path
            image = image.add_local_file(local_path, dest, copy=True)
        markers[marker_path(dest)] = content_digest(local_path)

    if markers:
        # Write the same digests ModalCommandRunner._rsync_up computes, so the
        # post-boot sync recognises the baked content and transfers nothing.
        # Without these the wheel would be re-sent through an 8 KB argv
        # channel, which is precisely what the bake exists to avoid.
        script = ' && '.join(
            f'mkdir -p {shlex.quote(os.path.dirname(path))} && '
            f'printf %s {shlex.quote(digest)} > {shlex.quote(path)}'
            for path, digest in sorted(markers.items()))
        image = image.run_commands(script)
    return image


#: The container runs as root, so ``~`` is ``/root``. Two things force us to
#: expand it ourselves rather than let a shell do it: Modal's
#: ``add_local_dir``/``add_local_file`` reject a non-absolute ``remote_path``
#: outright, and every path this module puts in a command is ``shlex.quote``d,
#: which stops the remote shell expanding a leading ``~`` and would silently
#: create a directory literally named ``~``.
REMOTE_HOME = '/root'


def abs_remote_path(path: str) -> str:
    """Expand a leading ``~`` against the container's home."""
    if path == '~':
        return REMOTE_HOME
    if path.startswith('~/'):
        return f'{REMOTE_HOME}/{path[2:]}'
    return path


def marker_path(remote_path: str) -> str:
    """Path of the sync marker that records what content is already present."""
    return f'{abs_remote_path(remote_path).rstrip("/")}.sky_modal_sync'


def content_digest(local_path: str) -> str:
    """Deterministic digest of a file or directory's contents.

    Deliberately content-based rather than archive-based: a tar's bytes vary
    with mtime, so a tar digest could never match between the image bake and a
    later comparison.
    """
    hasher = hashlib.sha256()
    local_path = os.path.expanduser(local_path)
    if os.path.isfile(local_path):
        hasher.update(b'file:')
        with open(local_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    elif os.path.isdir(local_path):
        hasher.update(b'dir:')
        for root, dirs, files in os.walk(local_path):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(root, name)
                hasher.update(os.path.relpath(path, local_path).encode())
                try:
                    with open(path, 'rb') as f:
                        hasher.update(f.read())
                except OSError:
                    continue
    return hasher.hexdigest()


def file_mounts_digest(file_mounts: Optional[List[Dict[str, str]]]) -> str:
    """Content digest of the baked file set.

    Part of the deployment's shape: if the wheel or a credential changed, the
    running container is stale and must not be reused.
    """
    hasher = hashlib.sha256()
    for mount in sorted(file_mounts or [], key=lambda m: m['RemotePath']):
        local_path = os.path.expanduser(mount['LocalPath'])
        hasher.update(mount['RemotePath'].encode())
        if os.path.isfile(local_path):
            with open(local_path, 'rb') as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        elif os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    hasher.update(os.path.relpath(path, local_path).encode())
                    try:
                        with open(path, 'rb') as f:
                            hasher.update(f.read())
                    except OSError:
                        continue
    return hasher.hexdigest()[:32]


def compute_spec_hash(node_config: Dict[str, Any],
                      file_mounts: Optional[List[Dict[str, str]]]) -> str:
    """Stable hash of everything that defines the deployment's shape."""
    shape = {
        'gpu': node_config.get('Gpu'),
        'cpu': node_config.get('Cpu'),
        'memory': node_config.get('Memory'),
        'named_image': node_config.get('NamedImage'),
        'docker_image': node_config.get('DockerImage'),
        'port': node_config.get('Port', DEFAULT_PORT),
        'routing_region': node_config.get('RoutingRegion'),
        'compute_region': node_config.get('ComputeRegion'),
        'files': file_mounts_digest(file_mounts),
    }
    return hashlib.sha256(json.dumps(shape,
                                     sort_keys=True).encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Server definition and deployment
# ---------------------------------------------------------------------------


def _make_server_class(app, image, node_config: Dict[str, Any],
                       secrets: List[Any], volumes: Dict[str, Any]):
    """Define and register the Server class on ``app``.

    The class is defined here, inside a library function, so ``serialized=True``
    is required -- Modal cannot ship a module it cannot import.
    """
    modal = _modal()
    port = int(node_config.get('Port') or DEFAULT_PORT)
    startup_timeout = int(node_config['StartupTimeout'])

    kwargs: Dict[str, Any] = {
        'image': image,
        'port': port,
        'name': SERVER_NAME,
        'serialized': True,
        'startup_timeout': startup_timeout,
        # A SkyPilot node must EXIST before any request arrives, because
        # SkyPilot execs its setup into it. A Server with no autoscaler hints
        # sits at zero replicas until traffic, leaving nothing to exec into
        # (verified: empty container list). Hence min_containers=1.
        #
        # This is NOT `max_containers=1`, which Modal's docs warn "will prevent
        # Modal from bringing up a replacement to gracefully shift traffic
        # during a rolling redeployment". Leaving max_containers unset keeps
        # that rolling-redeploy property while still pinning one warm node.
        # `target_concurrency` stays UNSET so the singleton does not autoscale.
        'min_containers': 1,
        # TODO(modal-auth): open endpoint. This is the current posture, not a
        # settled one -- proxy-token auth is the intended follow-up. Flagged
        # rather than shipped silently.
        'unauthenticated': True,
    }
    if node_config.get('Gpu'):
        kwargs['gpu'] = node_config['Gpu']
    else:
        # `nonpreemptible` is a CPU/memory-only knob. Modal rejects it outright
        # for any GPU workload -- `InvalidError: Non-preemptible is not
        # supported for GPU workloads` -- so sending it unconditionally failed
        # 100% of GPU launches.
        #
        # This does NOT make GPU nodes on-demand. Modal GPU Functions are
        # *always* preemptible and cannot be made otherwise: the docs state
        # plainly that "the `nonpreemptible` parameter is not supported for GPU
        # Functions", and the only on-demand lifecycle knob in the SDK
        # (`modal.SchedulerPlacement(spot=False)` -> `_lifecycle='on-demand'`)
        # is experimental and is not accepted by `@app.server()`, which exposes
        # `nonpreemptible` alone. A Server is a Function, not a Sandbox, so the
        # Sandbox carve-out ("not subject to preemption ... except where a gpu
        # requirement is specified") does not apply either.
        #
        # SPOT_INSTANCE therefore stays unsupported in `sky/clouds/modal.py` in
        # the sense that the *user cannot request* spot -- not in the sense that
        # GPU nodes are on-demand. See the note in `instance.py`.
        kwargs['nonpreemptible'] = True
    if node_config.get('Cpu') is not None:
        kwargs['cpu'] = node_config['Cpu']
    if node_config.get('Memory') is not None:
        kwargs['memory'] = node_config['Memory']
    if node_config.get('RoutingRegion'):
        kwargs['routing_region'] = node_config['RoutingRegion']
    if node_config.get('ComputeRegion'):
        kwargs['compute_region'] = node_config['ComputeRegion']
    if secrets:
        kwargs['secrets'] = secrets
    if volumes:
        kwargs['volumes'] = volumes

    baked = bool(node_config.get('NamedImage'))

    @app.server(**kwargs)
    class SkyPilotNode:  # pylint: disable=unused-variable
        """A SkyPilot node whose readiness is the workload's readiness."""

        @modal.enter()
        def wait_for_workload(self):
            # Deliberately does NOT bind the port itself.
            #
            # Modal's health check is TCP-level: it verifies that something is
            # *listening*, not that SkyRL is serving. A warming listener would
            # therefore report healthy while SkyRL is absent, permanently
            # blinding the platform watchdog -- which is the single main reason
            # Servers were chosen over Sandbox tunnels. So we wait for the real
            # workload instead, and buy the time with a generous
            # startup_timeout.
            import socket  # pylint: disable=import-outside-toplevel
            import time as _time  # pylint: disable=import-outside-toplevel

            # Make cold-vs-baked visible: a prebaked image reaches SETUP in
            # ~100s, a cold one takes ~10 min, and a slow start is otherwise
            # indistinguishable from a hang.
            print(
                f'[skypilot] node booting (image={"baked" if baked else "cold"}), '
                f'waiting up to {startup_timeout}s for the workload to bind '
                f'port {port}',
                flush=True)
            started = _time.time()
            deadline = started + startup_timeout - 15
            last_log = 0.0
            while _time.time() < deadline:
                sock = socket.socket()
                sock.settimeout(2)
                try:
                    sock.connect(('127.0.0.1', port))
                    elapsed = _time.time() - started
                    print(
                        f'[skypilot] workload is listening on port {port} '
                        f'after {elapsed:.0f}s',
                        flush=True)
                    return
                except OSError:
                    pass
                finally:
                    sock.close()
                now = _time.time()
                if now - last_log > 60:
                    print(
                        f'[skypilot] still waiting for port {port} '
                        f'({now - started:.0f}s elapsed)',
                        flush=True)
                    last_log = now
                _time.sleep(2)
            raise RuntimeError(
                f'[skypilot] workload did not bind port {port} within '
                f'{startup_timeout}s; Modal will replace this container.')

    return SkyPilotNode


def deploy_server(cluster_name_on_cloud: str, node_config: Dict[str, Any],
                  file_mounts: Optional[List[Dict[str, str]]],
                  environment_name: Optional[str]) -> Tuple[str, str]:
    """Deploy (or redeploy) the cluster's Server. Returns (app_id, spec_hash)."""
    modal = _modal()
    app_name = app_name_for_cluster(cluster_name_on_cloud)
    spec_hash = compute_spec_hash(node_config, file_mounts)

    image = build_image(node_config.get('NamedImage'),
                        node_config.get('DockerImage'), file_mounts)

    secrets = []
    env_secret = get_modal_env_secret()
    if env_secret is not None:
        secrets.append(env_secret)

    volumes = get_modal_volume_mounts(node_config.get('ModalVolumes', []))

    app = modal.App(app_name)
    _make_server_class(app, image, node_config, secrets, volumes)

    tags = {
        TAG_CLUSTER: cluster_name_on_cloud,
        TAG_SPEC: spec_hash,
    }
    max_lifetime_s = node_config.get('MaxLifetimeSeconds')
    if max_lifetime_s:
        tags[TAG_LIFETIME_DEADLINE] = str(
            int(time.time()) + int(max_lifetime_s))
    app.deploy(environment_name=environment_name)
    # Tags can only be set on a running App, so this necessarily follows the
    # deploy. run_instances() reads them back to enforce the shape guard.
    app.set_tags(tags)
    logger.info(f'Deployed Modal Server app {app_name} '
                f'(app_id={app.app_id}, spec={spec_hash}).')
    return app.app_id, spec_hash


def get_server(cluster_name_on_cloud: str,
               environment_name: Optional[str] = None):
    """Return the hydrated Server for a cluster, or None."""
    modal = _modal()
    app_name = app_name_for_cluster(cluster_name_on_cloud)
    try:
        server = modal.Server.from_name(app_name,
                                        SERVER_NAME,
                                        environment_name=environment_name)
        server.hydrate()
        return server
    except Exception as exc:  # pylint: disable=broad-except
        not_found = getattr(getattr(modal, 'exception', None), 'NotFoundError',
                            None)
        if not_found is not None and isinstance(exc, not_found):
            return None
        # An App that exists but has been stopped also lands here.
        logger.debug(f'No live Modal Server for {cluster_name_on_cloud}: {exc}')
        return None


def get_server_url(server) -> Optional[str]:
    try:
        return server.get_url()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug(f'Failed to read Modal Server URL: {exc}')
        return None


def get_app_tags(cluster_name_on_cloud: str,
                 environment_name: Optional[str] = None) -> Dict[str, str]:
    """Read an App's tags.

    Keyed on the App NAME: ``App.lookup`` resolves names, not app ids, and
    silently hands back a fresh empty App if given an id -- which would make
    the shape guard read every deployment as untagged and reject it.
    """
    modal = _modal()
    try:
        app = modal.App.lookup(app_name_for_cluster(cluster_name_on_cloud),
                               environment_name=environment_name,
                               create_if_missing=False)
        return dict(app.get_tags() or {})
    except Exception:  # pylint: disable=broad-except
        return {}


# ---------------------------------------------------------------------------
# Container discovery / lifecycle (CLI-backed)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _modal_cli_argv() -> List[str]:
    """Argv prefix for the Modal CLI, resolved without relying on ``PATH``.

    A bare ``['modal', ...]`` assumes the CLI is on ``PATH``, which is not safe:
    the API server can run under a different environment than the one that
    installed Modal. When that assumption breaks, ``subprocess.run`` raises
    ``FileNotFoundError`` -- and on the teardown path that means a *leaked,
    still-billing GPU container*, on the one path that must never fail.

    ``modal`` ships a ``__main__``, so the interpreter that can already import
    it can always invoke its CLI. Prefer that; fall back to ``PATH`` only when
    the module is not importable here.
    """
    if importlib.util.find_spec('modal') is not None:
        return [sys.executable, '-m', 'modal']
    return ['modal']


def _run_modal_cli(args: List[str],
                   timeout: int = 180) -> subprocess.CompletedProcess:
    argv = [*_modal_cli_argv(), *args]
    try:
        return subprocess.run(argv,
                              capture_output=True,
                              text=True,
                              timeout=timeout,
                              check=False)
    except FileNotFoundError as e:
        # Surface as a normal non-zero result so callers apply their own policy
        # (``stop_app`` raises loudly; discovery degrades) instead of dying on
        # an opaque OSError from deep inside teardown.
        return subprocess.CompletedProcess(argv,
                                           returncode=127,
                                           stdout='',
                                           stderr=f'Modal CLI not found: {e}')


def list_container_ids(app_id: str) -> List[str]:
    """Container (task) IDs for an App.

    Modal exposes no *public* SDK call that lists a non-Sandbox App's
    containers, so this shells out to the CLI. Same reason ``ModalCommandRunner``
    shells out per command -- it is a gap in Modal's public surface, not an
    oversight here.
    """
    proc = _run_modal_cli(['container', 'list', '--app-id', app_id, '--json'])
    if proc.returncode != 0:
        # An empty list here is indistinguishable from "all containers gone",
        # which is how `wait_for_containers_gone` reads it. Say so loudly, so a
        # still-billing container is never silently reported as torn down.
        logger.warning(
            f'Could not list containers for Modal app {app_id} '
            f'(exit {proc.returncode}): {proc.stderr.strip()}. '
            'Treating as empty; verify manually that nothing is still running.')
        return []
    try:
        rows = json.loads(proc.stdout or '[]')
    except json.JSONDecodeError:
        logger.warning(
            f'Unparseable container list for Modal app {app_id}; treating as '
            'empty. Verify manually that nothing is still running.')
        return []
    return [r['container_id'] for r in rows if r.get('container_id')]


def wait_for_container(app_id: str, timeout: int = 900) -> Optional[str]:
    """Wait until the App has a container we can exec into.

    A container is exec-able while ``@modal.enter()`` is still running, i.e.
    long before Modal reports it ready -- which is exactly what lets SkyPilot
    run its setup and bind the port that readiness depends on.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        containers = list_container_ids(app_id)
        if containers:
            return containers[0]
        time.sleep(4)
    return None


def stop_app(app_id: str) -> None:
    """Stop an App, terminating its containers.

    ``AUTOSTOP`` and ``AUTO_TERMINATE`` are unsupported on this cloud, so this
    is the only thing that reliably ends billing. Teardown correctness carries
    more weight here than on a cloud with a platform-side backstop.
    """
    proc = _run_modal_cli(['app', 'stop', '--yes', app_id], timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f'Failed to stop Modal app {app_id}: {proc.stderr.strip()}')


def wait_for_containers_gone(app_id: str, timeout: int = 180) -> List[str]:
    """Poll until the App has no containers. Returns any stragglers."""
    deadline = time.time() + timeout
    remaining = list_container_ids(app_id)
    while remaining and time.time() < deadline:
        time.sleep(5)
        remaining = list_container_ids(app_id)
    return remaining


def app_id_for_cluster(cluster_name_on_cloud: str,
                       environment_name: Optional[str] = None) -> Optional[str]:
    modal = _modal()
    app_name = app_name_for_cluster(cluster_name_on_cloud)
    try:
        app = modal.App.lookup(app_name,
                               environment_name=environment_name,
                               create_if_missing=False)
        return app.app_id
    except Exception:  # pylint: disable=broad-except
        return None


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

READY = 'ready'
WARMING = 'warming'
UNREACHABLE = 'unreachable'


def probe_endpoint(url: str, timeout: int = 20) -> str:
    """Classify the Server endpoint's state.

    A booting Server **rejects with 503 rather than queuing**, so 503 means
    "warming", not "broken". Treating it as failure would abort every launch.
    """
    proc = subprocess.run([
        'curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '-m',
        str(timeout), url
    ],
                          capture_output=True,
                          text=True,
                          check=False)
    code = (proc.stdout or '').strip()
    if code == '503':
        return WARMING
    if code and code[0] in '23':
        return READY
    if code in ('', '000'):
        return UNREACHABLE
    return WARMING


# ---------------------------------------------------------------------------
# Log relay (conserves evsys-enterprise #88)
# ---------------------------------------------------------------------------


def stream_server_logs(server,
                       sink,
                       stop_after: Optional[float] = None) -> None:
    """Relay the container's own logs into a caller-provided sink.

    Conserves evsys-enterprise #88, which shelled out to `modal app logs`;
    ``Server.logs.stream()`` is the cleaner equivalent.

    Deliberately does NOT deduplicate. #88's relay dedups unconditionally,
    which is correct against `modal app logs` (it replays) but WRONG here:
    ``stream()`` resumes rather than replaying, so a repeated line is a real
    repeated line and dropping it loses data.

    Cancellation uses the finite ``timeout=`` rather than killing a subprocess,
    since there is no subprocess to kill.
    """
    try:
        kwargs = {} if stop_after is None else {'timeout': stop_after}
        for entry in server.logs.stream(**kwargs):
            text = getattr(entry, 'data', None) or str(entry)
            sink(text.rstrip('\n'))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug(f'Modal log relay ended: {exc}')


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------


def get_modal_volume_mounts(
        volume_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build Modal Volume mount objects for the Server decorator."""
    modal = _modal()
    volumes = {}
    for spec in volume_specs or []:
        volume = modal.Volume.from_name(
            spec['VolumeNameOnCloud'],
            environment_name=spec.get('EnvironmentName'),
            create_if_missing=False)
        sub_path = spec.get('SubPath')
        if sub_path:
            volume = volume.with_mount_options(sub_path=sub_path)
        volumes[spec['Path']] = volume
    return volumes


# ---------------------------------------------------------------------------
# Exec plumbing shared with ModalCommandRunner
# ---------------------------------------------------------------------------


def argv_budget() -> int:
    """Bytes of command text a single exec may carry."""
    return _ARGV_LIMIT_BYTES - _ARGV_SAFETY_MARGIN


def drain_guard(command: str, sentinel: str) -> str:
    """Wrap a remote command so truncation is detectable, not silent.

    Appends the exit code and a sentinel, then lingers so the output stream
    finishes draining before the process exits (see _DRAIN_TRUNCATION_LIMIT).
    """
    return (f'{{ {command}; }}; __sky_rc=$?; '
            f'echo "{sentinel}:$__sky_rc"; '
            f'sleep {_DRAIN_DELAY_SECONDS}; exit $__sky_rc')


def split_sentinel(output: str,
                   sentinel: str) -> Tuple[str, Optional[int], bool]:
    """Split runner output on the sentinel.

    Returns (payload, returncode, complete). ``complete`` is False when the
    sentinel is missing, which means the 8192-char drain race ate the tail --
    the caller must retry rather than trust a partial result.
    """
    marker = f'{sentinel}:'
    idx = output.rfind(marker)
    if idx < 0:
        return output, None, False
    payload = output[:idx]
    rest = output[idx + len(marker):].strip().split()
    try:
        returncode = int(rest[0]) if rest else None
    except ValueError:
        returncode = None
    return payload, returncode, returncode is not None


def encode_chunks(data: bytes, chunk_size: int) -> List[str]:
    """Base64 the payload and split it into argv-sized pieces."""
    encoded = base64.b64encode(data).decode()
    return [
        encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)
    ]


def quote(value: str) -> str:
    return shlex.quote(value)
