"""Daytona GPU sandbox provisioning.

A SkyPilot cluster on Daytona is **one sandbox**: a container with up to 8
GPUs attached, reached over Daytona's SSH gateway.

HTTPS only, on purpose
----------------------
Daytona *has* SSH: ``POST /sandbox/{id}/ssh-access`` mints a token and
``ssh <token>@ssh.app.daytona.io`` connects. This provisioner deliberately does
not use it, because port 22 is blocked by the egress proxy of a locked agent
sandbox -- which is precisely why SkyPilot can rent Modal from inside one but
not Verda. Every call here is HTTPS to ``app.daytona.io`` /
``proxy.app.daytona.io``, so an agent running in a locked sandbox can rent
Daytona GPUs on the same terms it rents Modal.

The transport is :class:`~sky.utils.command_runner.DaytonaCommandRunner` over
``POST /toolbox/{id}/process/execute``, plus Daytona's binary-safe file
endpoints for transfers. Measured live: no argv cap (1 MB command fine), no
output truncation (1 MB of stdout fine), exit codes propagate, and a 3 MiB blob
round-trips with matching sha256. All four are the opposite of Modal's exec
channel, so none of Modal's chunking, staging or truncation-recovery machinery
is needed here. The one shared limitation is that **stderr is merged into
stdout** and cannot be separated.

One sandbox, one node
---------------------
``H100:2`` is two GPUs in ONE container, never two nodes, so MULTI_NODE is
unsupported. That is also the property the pools model wants: both GPUs share a
filesystem, so a trainer/sampler pair placed here is genuinely adjacent.

Capability constraints, all reflected in ``sky/clouds/daytona.py``
------------------------------------------------------------------
* ``STOP`` unsupported -- a GPU sandbox is created ``autoDeleteInterval: 0``
  and is *deleted* when it stops; its filesystem does not survive. There is no
  stopped cluster to come back to.
* ``AUTOSTOP`` / ``AUTO_TERMINATE`` unsupported in SkyPilot's sense, but
  Daytona enforces its own wall-clock ``ttlMinutes`` server-side regardless of
  state. That deadline is the teardown that survives the API server dying, and
  it is always set.
* ``IMAGE_ID`` is supported only as a docker image, and it travels as
  ``buildInfo.dockerfileContent`` -- ``POST /sandbox`` has no ``image`` field.
* ``SPOT_INSTANCE`` **is** supported, unlike Modal. It is real preemption: no
  warning, no webhook, and a create with no spare capacity fails immediately
  rather than queueing.
"""

from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.daytona import daytona_utils
from sky.utils import command_runner
from sky.utils import resources_utils
from sky.utils import status_lib
from sky.utils import ux_utils

PROVIDER_NAME = 'daytona'

logger = sky_logging.init_logger(__name__)


def _head_of(sandboxes: List[Dict[str, Any]]) -> Optional[str]:
    if not sandboxes:
        return None
    # Deterministic: oldest sandbox is the head, so a racing double-create
    # cannot flip which one SkyPilot talks to between calls.
    ordered = sorted(sandboxes, key=lambda s: str(s.get('createdAt') or ''))
    return str(ordered[0]['id'])


def run_instances(region: str, cluster_name: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    del cluster_name  # unused
    if config.count != 1:
        with ux_utils.print_exception_no_traceback():
            raise RuntimeError(
                'Daytona only supports single-node clusters. A Daytona '
                'sandbox is one container, so a multi-GPU request like '
                'H100:2 is two GPUs on that one node, not two nodes.')

    node_config = dict(config.node_config)
    existing = daytona_utils.find_sandboxes(cluster_name_on_cloud)
    head = _head_of(existing)
    if head is not None:
        record = daytona_utils.get_sandbox(head) or {}
        logger.info(f'Cluster {cluster_name_on_cloud} already has Daytona '
                    f'sandbox {head} (state '
                    f'{record.get("state", "unknown")!r}).')
        if str(record.get('state') or '').lower() not in ('started',):
            daytona_utils.wait_started(
                head, int(node_config.get('StartupTimeout') or 900))
        return common.ProvisionRecord(provider_name=PROVIDER_NAME,
                                      cluster_name=cluster_name_on_cloud,
                                      region=region,
                                      zone=None,
                                      head_instance_id=head,
                                      resumed_instance_ids=[],
                                      created_instance_ids=[])

    node_config['Region'] = region
    # Free preflight: quota is readable, stock is not. This turns "the org has
    # no room" from a 400 mid-provision into a refusal before anything exists.
    quota_problem = daytona_utils.check_gpu_quota(
        region, int(node_config.get('Gpu') or 0))
    if quota_problem is not None:
        with ux_utils.print_exception_no_traceback():
            raise RuntimeError(quota_problem)
    # A snapshot carries the machine shape, so a mismatched one would hand
    # back a different node than was planned AND priced. Also free.
    snapshot = node_config.get('Snapshot')
    if snapshot:
        shape_problem = daytona_utils.check_snapshot_shape(
            snapshot, node_config)
        if shape_problem is not None:
            with ux_utils.print_exception_no_traceback():
                raise RuntimeError(shape_problem)
        logger.info(f'Booting {cluster_name_on_cloud} from baked snapshot '
                    f'{snapshot} — SkyRL is already installed, so SETUP takes '
                    'its fast path.')
    sandbox_id = daytona_utils.create_sandbox(cluster_name_on_cloud,
                                              node_config)
    logger.info(f'Created Daytona sandbox {sandbox_id} for '
                f'{cluster_name_on_cloud}; waiting for it to start.')
    try:
        daytona_utils.wait_started(
            sandbox_id, int(node_config.get('StartupTimeout') or 900))
    except Exception:
        # The sandbox is already billing. Anything that goes wrong between
        # create and "started" must not leave it behind -- this is the failure
        # path, which is the one that leaks a GPU.
        logger.warning(f'Daytona sandbox {sandbox_id} did not start; deleting '
                       'it rather than leaving a GPU billing.')
        try:
            daytona_utils.delete_sandbox(sandbox_id)
        except Exception:  # pylint: disable=broad-except
            logger.warning(f'Could not delete {sandbox_id}; its ttlMinutes '
                           'deadline is the backstop.')
        raise
    # Daytona's stock images are root-without-sudo and have no ~/.ssh; both
    # break SkyPilot's setup commands. Same teardown-on-failure rule applies.
    try:
        daytona_utils.bootstrap_node(
            sandbox_id, node_config.get('DomainAllowList'))
    except Exception:
        logger.warning(f'Daytona sandbox {sandbox_id} failed bootstrap; '
                       'deleting it rather than leaving a GPU billing.')
        try:
            daytona_utils.delete_sandbox(sandbox_id)
        except Exception:  # pylint: disable=broad-except
            pass
        raise
    return common.ProvisionRecord(provider_name=PROVIDER_NAME,
                                  cluster_name=cluster_name_on_cloud,
                                  region=region,
                                  zone=None,
                                  head_instance_id=sandbox_id,
                                  resumed_instance_ids=[],
                                  created_instance_ids=[sandbox_id])


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    # run_instances() already blocked on `started`, which is the only
    # readiness this provisioner has: the workload SkyPilot is about to install
    # is what makes the box useful, and it installs over SSH from here.
    del region, cluster_name_on_cloud, state  # unused


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    del cluster_name_on_cloud, provider_config, worker_only  # unused
    with ux_utils.print_exception_no_traceback():
        raise NotImplementedError(
            'Stopping is not supported on Daytona: a GPU sandbox is created '
            'ephemeral (autoDeleteInterval=0) and is DELETED when it stops, '
            'with its filesystem discarded. Use `sky down` -- there is no '
            'stopped cluster to restart.')


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Delete every sandbox belonging to the cluster, and prove they are gone.

    Nothing else reaps a leaked Daytona GPU except its ``ttlMinutes``
    deadline, so this asserts its post-condition instead of assuming the
    DELETE landed.
    """
    del provider_config  # unused
    if worker_only:
        return
    sandboxes = daytona_utils.find_sandboxes(cluster_name_on_cloud)
    if not sandboxes:
        logger.debug(f'No Daytona sandbox for {cluster_name_on_cloud}; '
                     'nothing to terminate.')
        return
    failures: List[str] = []
    for sandbox in sandboxes:
        sandbox_id = str(sandbox['id'])
        try:
            daytona_utils.delete_sandbox(sandbox_id)
        except Exception as e:  # pylint: disable=broad-except
            failures.append(f'{sandbox_id}: {e}')
            continue
        remaining = daytona_utils.wait_gone(sandbox_id)
        if remaining is not None:
            failures.append(f'{sandbox_id}: still {remaining}')
    if failures:
        with ux_utils.print_exception_no_traceback():
            raise RuntimeError(
                'Failed to terminate Daytona sandboxes for '
                f'{cluster_name_on_cloud}: {"; ".join(failures)}. Billing '
                'continues until they are gone (their ttlMinutes deadline is '
                'the only other backstop).')
    logger.info(f'Terminated {len(sandboxes)} Daytona sandbox(es) for '
                f'{cluster_name_on_cloud}.')


def get_cluster_info(
        region: str,
        cluster_name_on_cloud: str,
        provider_config: Optional[Dict[str, Any]] = None) -> common.ClusterInfo:
    """Cluster metadata.

    There is no per-node IP and no SSH port: every node is reached over HTTPS
    by sandbox id. ``internal_ip`` is ``127.0.0.1`` so SkyPilot's ray wiring
    stays inside the container, and ``ssh_port`` is None -- which is also why
    ``sky/provision/provisioner.py`` skips its SSH probe for this cloud.
    """
    del region  # unused
    provider_config = dict(provider_config or {})
    sandboxes = daytona_utils.find_sandboxes(cluster_name_on_cloud)
    head_instance_id = _head_of(sandboxes)

    instances: Dict[str, List[common.InstanceInfo]] = {}
    for sandbox in sandboxes:
        sandbox_id = str(sandbox['id'])
        instances[sandbox_id] = [
            common.InstanceInfo(
                instance_id=sandbox_id,
                internal_ip='127.0.0.1',
                external_ip=None,
                ssh_port=None,
                tags={
                    'sandbox_id': sandbox_id,
                    'spot': str(bool(sandbox.get('spot'))),
                },
                node_name=sandbox_id)
        ]

    return common.ClusterInfo(instances=instances,
                              head_instance_id=head_instance_id,
                              provider_name=PROVIDER_NAME,
                              provider_config=provider_config,
                              ssh_user='root',
                              custom_ray_options=_ray_sizing(sandboxes,
                                                             head_instance_id))


def _ray_sizing(sandboxes: List[Dict[str, Any]],
                head_instance_id: Optional[str]) -> Dict[str, Any]:
    """Tell ray the sandbox's real size, because the container lies.

    A Daytona sandbox is cgroup-limited, but ``/proc`` is the **host's**.
    Daytona documents this outright -- "Tools such as ``nproc``, ``free``,
    ``top``, ``htop``, ``/proc/cpuinfo``, and ``/proc/meminfo`` read host-level
    values and do not reflect sandbox resource limits", pointing at
    ``/sys/fs/cgroup/cpu.max`` and ``/sys/fs/cgroup/memory.max`` instead
    (https://www.daytona.io/docs/en/sandboxes.md). Measured on an 8 vCPU /
    32 GiB sandbox: ``nproc`` reports **192** and ``free -g`` reports
    **338 GiB**. Ray sizes its heap, object store and
    worker pool from exactly those numbers when it is not told otherwise, so
    it tries to reserve host-scale memory inside a container that does not
    have it and dies at startup -- surfacing as SkyPilot's
    ``RuntimeError: Failed to start GCS`` with an empty error file, which
    names neither the cause nor the resource.

    So the sizes come from the sandbox record instead. The split (70% heap,
    20% object store) leaves headroom for the workload itself, which is the
    thing actually meant to use this box.
    """
    record = None
    for sandbox in sandboxes:
        if str(sandbox.get('id')) == head_instance_id:
            record = sandbox
            break
    if record is None:
        return {}
    cpus = int(record.get('cpu') or 0)
    memory_gib = int(record.get('memory') or 0)
    if not cpus or not memory_gib:
        return {}
    total_bytes = memory_gib * (1024 ** 3)
    return {
        'num-cpus': cpus,
        'memory': int(total_bytes * 0.70),
        'object-store-memory': int(total_bytes * 0.20),
        # NOTE: --node-ip-address is NOT set here. It comes from the
        # SKYPILOT_RAY_NODE_IP env var the provisioner puts on the sandbox,
        # which SkyPilot's own ray-start template already expands
        # (`${SKYPILOT_RAY_NODE_IP:+--node-ip-address=...}`). Setting it in
        # both places rendered the flag TWICE. See daytona_utils.create_sandbox
        # for why loopback is the only address that works once a
        # domainAllowList is in force.
    }


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Dict[str, Any],
) -> List[command_runner.CommandRunner]:
    """Command runners for a Daytona cluster.

    SSH credentials are dropped rather than threaded into a runner that cannot
    use them: this transport is HTTPS by design (see the module docstring), and
    silently accepting a key would suggest otherwise.
    """
    del credentials
    region = (cluster_info.provider_config or {}).get('region', '')
    runners: List[command_runner.CommandRunner] = []
    if cluster_info.head_instance_id is not None:
        runners.append(
            command_runner.DaytonaCommandRunner(
                (cluster_info.head_instance_id, region)))
    for sandbox_id in cluster_info.instances:
        if sandbox_id == cluster_info.head_instance_id:
            continue
        runners.append(
            command_runner.DaytonaCommandRunner((sandbox_id, region)))
    return runners


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    del cluster_name, provider_config, retry_if_missing  # unused
    sandboxes = daytona_utils.find_sandboxes(cluster_name_on_cloud,
                                             include_dead=not non_terminated_only)
    statuses: Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                              Optional[str]]] = {}
    for sandbox in sandboxes:
        sandbox_id = str(sandbox['id'])
        state = str(sandbox.get('state') or '').lower()
        if state in daytona_utils.RUNNING_STATES:
            statuses[sandbox_id] = (status_lib.ClusterStatus.UP, None)
        elif state in ('creating', 'starting', 'restoring',
                       'pulling_snapshot'):
            statuses[sandbox_id] = (status_lib.ClusterStatus.INIT, None)
        elif state in daytona_utils.DEAD_STATES:
            # Name a preemption as a preemption. Without spotEvictedAt, "the
            # box died" and "Daytona took the GPU back" are the same state.
            evicted = sandbox.get('spotEvictedAt')
            reason = ((f'spot preemption at {evicted}' if evicted else None) or
                      sandbox.get('errorReason'))
            statuses[sandbox_id] = (None, reason)
        else:
            statuses[sandbox_id] = (status_lib.ClusterStatus.INIT, state)
    return statuses


def open_ports(cluster_name_on_cloud: str,
               ports: List[str],
               provider_config: Optional[Dict[str, Any]] = None) -> None:
    del cluster_name_on_cloud, ports, provider_config  # unused
    # Nothing to open: every port 1-65535 is already reachable through the
    # preview proxy, gated by a token rather than a firewall rule.


def cleanup_ports(cluster_name_on_cloud: str,
                  ports: List[str],
                  provider_config: Optional[Dict[str, Any]] = None) -> None:
    del cluster_name_on_cloud, ports, provider_config  # unused


def query_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    head_ip: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[common.Endpoint]]:
    """Report each requested port as a **signed** preview URL.

    Daytona has no public per-node IP; a served port is reached through its
    preview proxy. The ordinary preview URL carries its token in an
    ``x-daytona-preview-token`` header, and SkyPilot's endpoint model has
    nowhere to put one -- nor does the client on the other end (the ``tinker``
    client that talks to a SkyRL server takes a base URL and nothing else).

    A **signed** preview URL embeds the token in the URL itself, so the
    endpoint SkyPilot hands back is directly usable. That is the whole reason
    this cloud can serve at all. Falls back to the header-gated URL when
    signing is unavailable, so ``sky status --endpoint`` still shows something
    true rather than nothing.
    """
    del head_ip  # unused
    expires = int((provider_config or {}).get('signed_url_seconds') or 86400)
    sandboxes = daytona_utils.find_sandboxes(cluster_name_on_cloud)
    head = _head_of(sandboxes)
    if head is None:
        return {}
    import urllib.parse  # pylint: disable=import-outside-toplevel
    out: Dict[int, List[common.Endpoint]] = {}
    for port in sorted(resources_utils.port_ranges_to_set(ports)):
        try:
            url = daytona_utils.signed_preview_url(head, int(port), expires)
        except daytona_utils.DaytonaError as e:
            logger.debug(f'No signed preview URL for port {port} ({e}); '
                         'falling back to the header-gated one.')
            try:
                url, _ = daytona_utils.preview_url(head, int(port))
            except daytona_utils.DaytonaError as e2:
                logger.debug(f'No Daytona preview URL for port {port}: {e2}')
                continue
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname is None:
            continue
        out[int(port)] = [
            common.HTTPSEndpoint(host=parsed.hostname,
                                 port=parsed.port,
                                 path=parsed.path.lstrip('/'))
        ]
    return out
