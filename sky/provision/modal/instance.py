"""Modal Server provisioning.

A SkyPilot cluster on Modal is one deployed Modal **App** holding one
``@app.server()`` Server, whose single container is the node.

Why Servers and not Sandboxes
-----------------------------
Sandbox tunnels are immutable after ``Sandbox.create()`` -- ``open_ports()`` is
a documented no-op -- and there is no way to put a Sandbox behind Modal's
request proxy, which binds to deployed App objects. A replaced sandbox
therefore always means a new URL, so the Sandbox path structurally cannot offer
a stable endpoint. We also have field evidence: serving tunnels dropped
permanently under sustained load while the sandbox itself stayed healthy
(``benchmarking/spot-gpu/MODAL-TUNNEL-RCA.md`` in evsys-enterprise). A Server's
URL comes from ``modal.Server.get_url()`` and survives container replacement.

A STABLE URL IS NOT STABLE STATE
--------------------------------
This is the single most important thing to understand before relying on it.
Modal health-checks the port and **terminates and replaces containers that fail
too many consecutive checks**. The replacement keeps the URL, but it destroys
the ray cluster and every scrap of SkyPilot runtime state on the node --
while SkyPilot goes on believing it still has a cluster with a running job.

Recovery is therefore **not** handled here and cannot be: it stays at the SDK
level, via durable checkpointing plus session resume. Do not mistake a
reachable endpoint for a live job.

Node identity
-------------
A Server is a singleton container, so ``gpu="H100:2"`` is two GPUs on ONE node,
never two nodes. ``ModalCommandRunner.node_id`` keys on the App for exactly
this reason, giving the pools model (``WorkerRegistry``, ``cluster_name`` +
``host`` -> ``node_id``) a single stable node identity per cluster.

Capability constraints, all reflected in ``sky/clouds/modal.py``
----------------------------------------------------------------
* ``IMAGE_ID`` unsupported -- our prebake Named Image cannot ride SkyPilot's
  ``image_id`` field, so it travels as its own ``NamedImage`` node_config key
  and is resolved with ``Image.from_name()``.
* ``SPOT_INSTANCE`` unsupported -- Modal candidates are on-demand only, which
  the Server decorator enforces with ``nonpreemptible=True``.
* ``AUTOSTOP`` / ``AUTO_TERMINATE`` unsupported -- ``max_lifetime_s`` is the
  ONLY teardown backstop, so ``terminate_instances`` correctness carries more
  weight here than on clouds with a platform-side sweeper.
"""

from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.modal import modal_utils
from sky.utils import command_runner
from sky.utils import common_utils
from sky.utils import status_lib
from sky.utils import ux_utils

PROVIDER_NAME = 'modal'

logger = sky_logging.init_logger(__name__)


def _running_containers(
        cluster_name_on_cloud: str,
        environment_name: Optional[str]) -> Tuple[Optional[str], List[str]]:
    app_id = modal_utils.app_id_for_cluster(cluster_name_on_cloud,
                                            environment_name)
    if app_id is None:
        return None, []
    return app_id, modal_utils.list_container_ids(app_id)


def run_instances(region: str, cluster_name: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    del cluster_name  # unused
    if config.count != 1:
        raise RuntimeError('Modal only supports single-node clusters. A Modal '
                           'Server is one container, so multi-GPU requests '
                           'like H100:2 are two GPUs on that one node.')

    environment_name = config.provider_config.get('environment_name')
    node_config = config.node_config
    file_mounts = node_config.get('FileMounts', [])
    spec_hash = modal_utils.compute_spec_hash(node_config, file_mounts)

    app_id, containers = _running_containers(cluster_name_on_cloud,
                                             environment_name)
    if app_id is not None and containers:
        # Conserves evsys-enterprise #87: refuse to reuse a deployment whose
        # shape no longer matches the request. This matters MORE here than it
        # did on the Sandbox path -- a Server's URL is derived from its name, so
        # a stale deployment answers on exactly the URL the caller expects and
        # looks correct while running the wrong GPU, image or file set.
        tags = modal_utils.get_app_tags(cluster_name_on_cloud, environment_name)
        existing = tags.get(modal_utils.TAG_SPEC)
        if existing != spec_hash:
            with ux_utils.print_exception_no_traceback():
                raise RuntimeError(
                    f'Modal app for cluster {cluster_name_on_cloud} is already '
                    f'deployed with a different shape (spec {existing} != '
                    f'requested {spec_hash}). Refusing to reuse it: the URL '
                    'would look correct while the node is wrong. Tear the '
                    'cluster down first.')
        logger.info(f'Cluster {cluster_name_on_cloud} already has a running '
                    f'Modal Server container ({containers[0]}).')
        return common.ProvisionRecord(provider_name=PROVIDER_NAME,
                                      cluster_name=cluster_name_on_cloud,
                                      region=region,
                                      zone=None,
                                      head_instance_id=containers[0],
                                      resumed_instance_ids=[],
                                      created_instance_ids=[])

    app_id, spec_hash = modal_utils.deploy_server(cluster_name_on_cloud,
                                                  node_config, file_mounts,
                                                  environment_name)
    container_id = modal_utils.wait_for_container(app_id)
    if container_id is None:
        raise RuntimeError(
            f'Modal Server for {cluster_name_on_cloud} did not start a '
            'container. A Server with min_containers=1 should always hold one; '
            'check `modal app logs` for an image build or scheduling failure.')
    logger.info(f'Modal Server container {container_id} is up for '
                f'{cluster_name_on_cloud}.')
    return common.ProvisionRecord(provider_name=PROVIDER_NAME,
                                  cluster_name=cluster_name_on_cloud,
                                  region=region,
                                  zone=None,
                                  head_instance_id=container_id,
                                  resumed_instance_ids=[],
                                  created_instance_ids=[container_id])


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    # run_instances() already waited for an exec-able container, which is the
    # only readiness this provisioner needs. It deliberately does NOT wait for
    # the Server to report READY: the container only becomes ready once the
    # workload binds the port, and the workload is what SkyPilot is about to
    # install over this very exec channel.
    del region, cluster_name_on_cloud, state  # unused


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    del cluster_name_on_cloud, provider_config, worker_only  # unused
    raise NotImplementedError(
        'stop_instances is not supported for Modal; use down instead.')


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Stop the cluster's App, terminating its container.

    ``AUTOSTOP``/``AUTO_TERMINATE`` are unsupported on Modal, so nothing else
    will ever clean this up: if this fails silently the GPU bills until someone
    notices. Hence the explicit post-condition check.
    """
    if worker_only:
        return
    environment_name = (provider_config or {}).get('environment_name')
    app_id = modal_utils.app_id_for_cluster(cluster_name_on_cloud,
                                            environment_name)
    if app_id is None:
        logger.debug(f'No Modal app for {cluster_name_on_cloud}; nothing to '
                     'terminate.')
        return
    try:
        modal_utils.stop_app(app_id)
    except Exception as e:  # pylint: disable=broad-except
        with ux_utils.print_exception_no_traceback():
            raise RuntimeError(
                f'Failed to terminate Modal app {app_id}: '
                f'{common_utils.format_exception(e, use_bracket=False)}') from e
    # `modal app stop` returns before the containers are actually gone, so poll
    # the post-condition rather than asserting it instantly. This check is the
    # whole teardown backstop on this cloud -- nothing else reaps a leaked
    # container, and a GPU node bills until it dies.
    remaining = modal_utils.wait_for_containers_gone(app_id)
    if remaining:
        with ux_utils.print_exception_no_traceback():
            raise RuntimeError(
                f'Modal app {app_id} still has containers {remaining} after '
                'stop. Billing continues until they are gone -- stop them with '
                f'`modal app stop {app_id}`.')
    logger.info(f'Terminated Modal app {app_id} for {cluster_name_on_cloud}.')


def get_cluster_info(
        region: str,
        cluster_name_on_cloud: str,
        provider_config: Optional[Dict[str, Any]] = None) -> common.ClusterInfo:
    del region  # unused
    environment_name = (provider_config or {}).get('environment_name')
    app_id, containers = _running_containers(cluster_name_on_cloud,
                                             environment_name)
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for container_id in containers:
        instances[container_id] = [
            common.InstanceInfo(
                instance_id=container_id,
                # There is no routable per-node IP: the node is reached through
                # `modal container exec`, and the workload through the Server
                # URL. 127.0.0.1 keeps SkyPilot's ray wiring in-container.
                internal_ip='127.0.0.1',
                external_ip=None,
                ssh_port=None,
                tags={'app_id': app_id or ''},
                node_name=container_id)
        ]
        if head_instance_id is None:
            head_instance_id = container_id

    provider_config = dict(provider_config or {})
    if app_id is not None:
        provider_config['app_id'] = app_id
    return common.ClusterInfo(instances=instances,
                              head_instance_id=head_instance_id,
                              provider_name=PROVIDER_NAME,
                              provider_config=provider_config,
                              ssh_user='root')


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Dict[str, Any],
) -> List[command_runner.CommandRunner]:
    """Command runners for a Modal cluster.

    SSH credentials are irrelevant here -- there is no SSH -- so they are
    dropped rather than threaded into a runner that cannot use them.
    """
    del credentials  # Modal has no SSH; nothing to authenticate with.
    app_id = (cluster_info.provider_config or {}).get('app_id', '')
    runners: List[command_runner.CommandRunner] = []
    if cluster_info.head_instance_id is not None:
        runners.append(
            command_runner.ModalCommandRunner(
                (cluster_info.head_instance_id, app_id)))
    for container_id in cluster_info.instances:
        if container_id == cluster_info.head_instance_id:
            continue
        runners.append(command_runner.ModalCommandRunner(
            (container_id, app_id)))
    return runners


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    del cluster_name, retry_if_missing  # unused
    environment_name = (provider_config or {}).get('environment_name')
    _, containers = _running_containers(cluster_name_on_cloud, environment_name)
    statuses: Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                              Optional[str]]] = {}
    for container_id in containers:
        statuses[container_id] = (status_lib.ClusterStatus.UP, None)
    if not containers and not non_terminated_only:
        return {}
    return statuses


def open_ports(cluster_name_on_cloud: str,
               ports: List[str],
               provider_config: Optional[Dict[str, Any]] = None) -> None:
    del cluster_name_on_cloud, ports, provider_config  # unused
    # A Server exposes exactly one port, fixed at deploy time by the decorator.
    # OPEN_PORTS_VERSION is LAUNCH_ONLY for this cloud.


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
    """Report the Server's endpoint.

    The endpoint exists as soon as the App is deployed, well before the
    workload answers on it. A booting Server **rejects with 503 rather than
    queuing**, so callers polling this endpoint must read 503 as "warming", not
    "broken" -- see ``modal_utils.probe_endpoint``.
    """
    del head_ip  # unused
    environment_name = (provider_config or {}).get('environment_name')
    server = modal_utils.get_server(cluster_name_on_cloud, environment_name)
    if server is None:
        return {}
    url = modal_utils.get_server_url(server)
    if not url:
        return {}
    import urllib.parse  # pylint: disable=import-outside-toplevel
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname is None:
        return {}
    endpoint = common.HTTPSEndpoint(host=parsed.hostname,
                                    port=parsed.port,
                                    path=parsed.path.lstrip('/'))
    requested = ports or []
    port_numbers = []
    for spec in requested:
        try:
            port_numbers.append(int(str(spec).split('-')[0]))
        except ValueError:
            continue
    if not port_numbers:
        port_numbers = [modal_utils.DEFAULT_PORT]
    return {port_numbers[0]: [endpoint]}
