"""Daytona cloud.

A cluster on Daytona is one **GPU sandbox**: a container with up to 8 NVIDIA
GPUs, reached over Daytona's SSH gateway. See
``sky/provision/daytona/instance.py`` for the provisioning model and
``sky/catalog/daytona_catalog.py`` for where the prices come from.

What is honestly encoded here rather than faked
-----------------------------------------------
* **No stopping.** A GPU sandbox is created ephemeral and is *deleted* when it
  stops, discarding its filesystem. There is no stopped cluster to restart, so
  ``STOP`` is unsupported rather than quietly mapped onto teardown.
* **No multi-node.** One sandbox is one container; ``H100:2`` is two GPUs on
  that single node.
* **Spot IS supported** -- real preemptible GPU capacity. Not the Modal
  situation. It is preemption without warning or webhook, and a create with no
  spare capacity fails immediately rather than queueing.
* **Ports are proxied, not opened.** Every port is already reachable through
  Daytona's preview proxy, but the URL is *authenticated*: it needs an
  ``x-daytona-preview-token`` header unless the sandbox is public. SkyPilot's
  endpoint model has nowhere to carry that header, so ``sky status --endpoint``
  gives a URL a browser will refuse without the token.
* **Region is advisory for GPUs.** Daytona documents that the requested
  target is ignored for GPU sandboxes because of scarcity.
* **Only NVIDIA H100 / H200 / RTX-PRO-6000 / RTX-5090 / RTX-4090.** No A100 --
  an A100 request finds no feasible resource here rather than silently landing
  on something else.
"""

import os
import re
import typing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sky import catalog
from sky import clouds
from sky import exceptions
from sky import skypilot_config
from sky.provision.daytona import daytona_utils
from sky.utils import annotations
from sky.utils import registry
from sky.utils import resources_utils
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky.utils import volume as volume_lib

_CREDENTIAL_FILE = daytona_utils.CREDENTIAL_FILE
_DEFAULT_REGION = 'us'
#: A CUDA-bearing public image. Daytona's stock snapshot is 1 vCPU / 1 GiB /
#: 1 GiB, which cannot host a SkyPilot runtime, so a GPU node needs a real one.
_DEFAULT_IMAGE = 'pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime'
#: Cold-pulling a CUDA image is the slow part of a Daytona launch.
_DEFAULT_STARTUP_TIMEOUT_S = 15 * 60
#: Wall-clock deadline Daytona enforces itself. AUTOSTOP/AUTO_TERMINATE are
#: unsupported here, so this is the teardown that survives the API server
#: dying. Never left unset by default.
_DEFAULT_TTL_MINUTES = 6 * 60
#: How long a minted SSH access token lasts. Re-minted on every
#: ``get_cluster_info()``, so this only bounds how stale a cached one can be.
_DEFAULT_SSH_ACCESS_MINUTES = 24 * 60


@registry.CLOUD_REGISTRY.register
class Daytona(clouds.Cloud):
    """Daytona cloud, backed by GPU sandboxes."""

    _REPR = 'Daytona'
    _MAX_CLUSTER_NAME_LEN_LIMIT = 60
    # yapf: disable
    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP:
            'A Daytona GPU sandbox is ephemeral: it is deleted when it stops '
            'and its filesystem is discarded. Use `sky down`.',
        clouds.CloudImplementationFeatures.MULTI_NODE:
            'A Daytona sandbox is one container; H100:2 is two GPUs on one '
            'node. Multi-node clusters are not supported.',
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            'Disk cloning is not supported on Daytona.',
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            'Custom disk tiers are not supported on Daytona.',
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            'Custom network tiers are not supported on Daytona.',
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            'Host controllers are not supported on Daytona.',
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            'High availability controllers are not supported on Daytona.',
        clouds.CloudImplementationFeatures.AUTO_TERMINATE:
            'SkyPilot-side auto-termination is not supported on Daytona; the '
            'sandbox carries its own server-side ttlMinutes deadline instead.',
        clouds.CloudImplementationFeatures.AUTOSTOP:
            'Autostop is not supported on Daytona: stopping a GPU sandbox '
            'deletes it. Use the sandbox ttlMinutes deadline.',
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            'Custom multiple network interfaces are not supported on Daytona.',
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            'Local disk requests are not supported on Daytona.',
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            'Cloud storage mounting is not supported on Daytona sandboxes.',
    }
    # yapf: enable

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.LAUNCH_ONLY

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: 'resources_lib.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del region  # unused
        unsupported = dict(cls._CLOUD_UNSUPPORTED_FEATURES)
        if resources is not None and not resources.accelerators:
            # Spot is GPU-only: Daytona rejects `spot: true` on a sandbox that
            # requests no GPUs.
            unsupported[clouds.CloudImplementationFeatures.SPOT_INSTANCE] = (
                'Daytona spot applies to GPU sandboxes only.')
        return unsupported

    @classmethod
    def max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None,
    ) -> List[clouds.Region]:
        assert zone is None, 'Daytona does not support zones.'
        del accelerators, zone, resources  # unused
        regions = catalog.get_region_zones_for_instance_type(
            instance_type, use_spot, 'daytona')
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> Iterator[None]:
        del region, num_nodes, instance_type, accelerators, use_spot  # unused
        yield None

    def instance_type_to_hourly_cost(self,
                                     instance_type: str,
                                     use_spot: bool,
                                     region: Optional[str] = None,
                                     zone: Optional[str] = None) -> float:
        return catalog.get_hourly_cost(instance_type,
                                       use_spot=use_spot,
                                       region=region,
                                       zone=zone,
                                       clouds='daytona')

    def accelerators_to_hourly_cost(self,
                                    accelerators: Dict[str, int],
                                    use_spot: bool,
                                    region: Optional[str] = None,
                                    zone: Optional[str] = None) -> float:
        # The instance price already carries the GPU: Daytona bills one
        # sandbox, not a host plus attached accelerators.
        del accelerators, use_spot, region, zone  # unused
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        del num_gigabytes  # unused
        return 0.0

    @classmethod
    def is_label_valid(cls, label_key: str,
                       label_value: str) -> Tuple[bool, Optional[str]]:
        key_regex = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9._-]{0,62})?$')
        value_regex = re.compile(r'^[a-zA-Z0-9._-]{0,63}$')
        if not key_regex.match(label_key):
            return False, (f'Invalid label key {label_key} for Daytona: keys '
                           'must be alphanumeric, dots, dashes or underscores, '
                           'up to 63 characters.')
        if not value_regex.match(label_value):
            return False, (f'Invalid label value {label_value} for Daytona: '
                           'values must be alphanumeric, dots, dashes or '
                           'underscores, up to 63 characters.')
        return True, None

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        return catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            disk_tier=disk_tier,
            local_disk=local_disk,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
            clouds='daytona')

    @classmethod
    def get_accelerators_from_instance_type(
        cls,
        instance_type: str,
    ) -> Optional[Dict[str, Union[int, float]]]:
        return catalog.get_accelerators_from_instance_type(instance_type,
                                                           clouds='daytona')

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls,
        instance_type: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        return catalog.get_vcpus_mem_from_instance_type(instance_type,
                                                        clouds='daytona')

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: resources_utils.ClusterName,
        region: clouds.Region,
        zones: Optional[List[clouds.Zone]],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Any]:
        del cluster_name, dryrun  # unused
        if num_nodes != 1:
            raise ValueError('Daytona only supports single-node clusters.')
        assert zones is None, 'Daytona does not support zones.'
        if volume_mounts:
            raise ValueError('Daytona clusters do not support volume mounts.')
        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)
        # pylint: disable=import-outside-toplevel
        from sky.catalog import daytona_catalog
        (gpu_type, gpu_count, cpu, memory,
         disk) = daytona_catalog.get_daytona_args_from_instance_type(
             resources.instance_type)
        # SkyPilot's own disk_size wins over the catalog default: the catalog
        # value exists to price a node, not to cap it.
        if resources.disk_size:
            disk = max(int(resources.disk_size), disk)

        def _daytona_config(key: str, default: Any) -> Any:
            return skypilot_config.get_effective_region_config(
                cloud='daytona',
                region=region.name,
                keys=(key,),
                default_value=default)

        return {
            'instance_type': resources.instance_type,
            'custom_resources': custom_resources,
            'region': region.name,
            'daytona_gpu_type': gpu_type,
            'daytona_gpu': gpu_count,
            'daytona_cpu': cpu,
            'daytona_memory': memory,
            'daytona_disk': disk,
            'daytona_use_spot': bool(resources.use_spot),
            'daytona_image': (resources.extract_docker_image() or
                              _daytona_config('image', _DEFAULT_IMAGE)),
            'daytona_snapshot': _daytona_config('snapshot', None),
            'daytona_domain_allow_list': _daytona_config(
                'domain_allow_list', None),
            'daytona_startup_timeout': _daytona_config(
                'startup_timeout', _DEFAULT_STARTUP_TIMEOUT_S),
            'daytona_ttl_minutes': _daytona_config('ttl_minutes',
                                                   _DEFAULT_TTL_MINUTES),
            'daytona_ssh_access_minutes': _daytona_config(
                'ssh_access_minutes', _DEFAULT_SSH_ACCESS_MINUTES),
        }

    def _get_feasible_launchable_resources(
        self, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            if not catalog.instance_type_exists(resources.instance_type,
                                                'daytona'):
                raise ValueError(
                    f'Invalid instance type: {resources.instance_type}')
            resources = resources.copy(accelerators=None,
                                       cpus=None,
                                       memory=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            resource_list = []
            for instance_type in instance_list:
                r = resources.copy(cloud=Daytona(),
                                   instance_type=instance_type,
                                   accelerators=None,
                                   cpus=None,
                                   memory=None)
                resource_list.append(r)
            return resource_list

        accelerators = resources.accelerators
        if accelerators is None:
            if resources.use_spot:
                # Spot is GPU-only on Daytona; a CPU-only spot ask has no
                # feasible resource here rather than silently becoming
                # on-demand.
                return resources_utils.FeasibleResources([], [], None)
            default_instance_type = Daytona.get_default_instance_type(
                cpus=resources.cpus,
                memory=resources.memory,
                disk_tier=resources.disk_tier,
                local_disk=resources.local_disk,
                region=resources.region,
                zone=resources.zone,
                use_spot=resources.use_spot,
                max_hourly_cost=resources.max_hourly_cost)
            if default_instance_type is None:
                return resources_utils.FeasibleResources([], [], None)
            return resources_utils.FeasibleResources(
                _make([default_instance_type]), [], None)

        assert len(accelerators) == 1, resources
        acc, acc_count = list(accelerators.items())[0]
        (instance_list,
         fuzzy_candidate_list) = catalog.get_instance_type_for_accelerator(
             acc,
             acc_count,
             use_spot=resources.use_spot,
             cpus=resources.cpus,
             memory=resources.memory,
             local_disk=resources.local_disk,
             region=resources.region,
             zone=resources.zone,
             max_hourly_cost=resources.max_hourly_cost,
             clouds='daytona')
        if instance_list is None:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        return resources_utils.FeasibleResources(_make(instance_list),
                                                 fuzzy_candidate_list, None)

    @classmethod
    def _check_compute_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        try:
            return daytona_utils.check_credentials()
        except Exception as e:  # pylint: disable=broad-except
            return False, (
                'Failed to check Daytona credentials. Set '
                f'{daytona_utils.API_KEY_ENV_VAR} or write '
                f'{_CREDENTIAL_FILE}. ({e})')

    def get_credential_file_mounts(self) -> Dict[str, str]:
        # The API key is the control-plane credential; it deliberately does NOT
        # travel to the node. A sandbox that can create sandboxes is a sandbox
        # that can spend the whole account.
        return {}

    def instance_type_exists(self, instance_type: str) -> bool:
        return catalog.instance_type_exists(instance_type, 'daytona')

    def validate_region_zone(self, region: Optional[str], zone: Optional[str]):
        return catalog.validate_region_zone(region, zone, clouds='daytona')

    @classmethod
    def regions(cls) -> List[clouds.Region]:
        return catalog.regions(clouds='daytona')

    @classmethod
    @annotations.lru_cache(scope='global', maxsize=1)
    def get_user_identities(cls) -> Optional[List[List[str]]]:
        try:
            who = daytona_utils.whoami()
            org = daytona_utils.organization_id()
        except Exception as e:  # pylint: disable=broad-except
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    f'Failed to get Daytona identity: {e}') from e
        name = who.get('name') or os.environ.get(
            daytona_utils.API_KEY_ENV_VAR, '')[:12]
        if org is None:
            return [[f'Daytona key {name}']]
        return [[f'Daytona organization {org}, key {name}'],
                [f'Daytona organization {org}']]
