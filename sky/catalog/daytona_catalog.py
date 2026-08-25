"""Daytona service catalog.

Daytona publishes fixed prices and exposes **no** forward pricing API (checked
live: ``/api/pricing``, ``/api/prices``, ``/api/billing/pricing`` and
``/api/organizations/{id}/pricing`` all answer 404). So this catalog is
synthesized from the published rate card rather than fetched, in the same shape
``modal_catalog`` uses, and it records where the numbers came from.

Two things here differ from Modal's catalog and both are Daytona facts, not
style:

* **Spot is real.** Daytona sells preemptible GPU capacity (``spot: true`` on
  ``POST /api/sandbox``), so every row carries a ``SpotPrice`` and
  ``SPOT_INSTANCE`` is a *supported* feature on this cloud.
* **CPU, RAM and disk are billed separately from the GPU, and they are not
  rounding error.** At $0.0162/GiB/hr the 192 GiB a single GPU is *allowed*
  costs $3.11/hr -- more than the H100 it hangs off. Pricing a Daytona node on
  its GPU rate alone would understate it by a factor at the top of the range,
  so the price of an instance type is computed over all four dimensions.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Union

from sky.adaptors import common as adaptors_common
from sky.catalog import common
from sky.clouds import cloud
from sky.utils import resources_utils
from sky.utils import ux_utils

pd = adaptors_common.LazyImport('pandas')

_CLOUD_NAME = 'daytona'
_DISPLAY_NAME = 'Daytona'

#: Daytona's regions. ``GET /api/regions`` returned only ``us`` on the account
#: this was built against; ``eu`` is documented. Both are listed because the
#: catalog must not silently forbid a region an org has been granted.
#:
#: Region is reported, never used to steer a GPU: the docs are explicit that
#: "the target/region requested for GPU sandboxes is ignored by default"
#: because of scarcity. A GPU request lands where there is capacity.
#:
#: A region an organization has not been granted answers **403 "Region eu is
#: not available to the organization for class container"** at create, and
#: SkyPilot's own fallback moves to the next candidate -- correct behaviour,
#: but it costs one wasted attempt, so the commonly-granted region is listed
#: first. Region grants are per-org and there is no catalog-time way to know
#: them; ``available-sandbox-classes`` answers it live at provision time.
_REGIONS = ('us', 'eu')
_DEFAULT_REGION = 'us'

# ---------------------------------------------------------------------------
# Rate card. Source: https://www.daytona.io/pricing -- taken 2026-08-24.
# ---------------------------------------------------------------------------

_GPU_PRICE_PER_HOUR = {
    'H200': 2.61,
    'H100': 2.27,
    'RTX-PRO-6000': 1.74,
    'RTX-5090': 0.74,
    'RTX-4090': 0.57,
}

_VCPU_PRICE_PER_HOUR = 0.0504
_MEMORY_GIB_PRICE_PER_HOUR = 0.0162
_DISK_GIB_PRICE_PER_HOUR = 0.000108

#: **Spot is quoted at the on-demand rate.** The published pricing page renders
#: GPU prices behind a Preemptible/On-demand toggle whose non-selected half is
#: client-rendered and absent from the served HTML, so only the on-demand
#: column is readable without a browser.
#:
#: Spot cannot be *dearer* than on-demand, so quoting the on-demand rate is a
#: safe upper bound. Inventing a plausible discount would make Daytona spot win
#: optimizer decisions it has not earned -- a catalog that lies downward is
#: exactly the failure a synthesized catalog must avoid. Set this below 1.0
#: only against a measured bill.
_SPOT_PRICE_MULTIPLIER = 1.0

_GPU_MEMORY_GIB = {
    'H200': 141,
    'H100': 80,
    'RTX-PRO-6000': 96,
    'RTX-5090': 32,
    'RTX-4090': 24,
}

#: Daytona takes up to 8 GPUs in one sandbox.
_GPU_COUNTS = {name: (1, 2, 4, 8) for name in _GPU_PRICE_PER_HOUR}

#: Ceilings Daytona enforces per GPU unit: "each GPU adds up to 16 vCPUs,
#: 192GB RAM, and 512GB disk". Asking beyond these is rejected at create.
_MAX_VCPUS_PER_GPU = 16
_MAX_MEMORY_GIB_PER_GPU = 192
_MAX_DISK_GIB_PER_GPU = 512

#: The shape a GPU node gets by default -- and the shape it is priced at. This
#: is a cost decision, not a detail: see the module docstring.
_DEFAULT_VCPUS_PER_GPU = 4
_DEFAULT_MEMORY_GIB_PER_GPU = 16
_DEFAULT_DISK_GIB_PER_GPU = 50

#: Non-GPU sandbox ceilings on the default tier (docs: CPU 1-4, memory 1-8,
#: disk 1-10). A higher org tier raises them, so these bound the *catalog*
#: rather than the platform.
_DEFAULT_CPU_VCPUS = 2
_DEFAULT_CPU_MEMORY_GIB = 4
_DEFAULT_CPU_DISK_GIB = 10
_MAX_CPU_ONLY_VCPUS = 4
_MAX_CPU_ONLY_MEMORY_GIB = 8

_VIRTUAL_INSTANCE_TYPE_PATTERN = re.compile(
    r'^(?P<vcpus>\d+(?:\.\d+)?)CPU--'
    r'(?P<memory>\d+(?:\.\d+)?)GB'
    r'(?:--(?P<accelerator>[\w-]+):(?P<count>\d+))?$')


def _format_resource_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f'{value:.12f}'.rstrip('0').rstrip('.')


def _canonical_accelerator(accelerator: str) -> Optional[str]:
    """Daytona's ``GpuType`` name for a caller's GPU string, or None.

    None means *Daytona does not sell this*, and every caller turns that into
    an empty result rather than a substitution. A100 is the case that matters:
    it is in common accelerator chains and Daytona has never sold one, so it
    must answer "no", not "here is an H100".
    """
    normalized = re.sub(r'[^a-z0-9]', '', accelerator.lower())
    for supported in _GPU_PRICE_PER_HOUR:
        if re.sub(r'[^a-z0-9]', '', supported.lower()) == normalized:
            return supported
    return None


def _price_per_hour(vcpus: float, memory_gib: float, disk_gib: float,
                    gpu_name: Optional[str], gpu_count: int) -> float:
    """All-in $/hr: GPU + vCPU + RAM + disk, every dimension Daytona bills."""
    price = (vcpus * _VCPU_PRICE_PER_HOUR +
             memory_gib * _MEMORY_GIB_PRICE_PER_HOUR +
             disk_gib * _DISK_GIB_PRICE_PER_HOUR)
    if gpu_name is not None:
        price += _GPU_PRICE_PER_HOUR[gpu_name] * gpu_count
    return price


class DaytonaInstanceType:
    """A Daytona sandbox shape, as a SkyPilot instance type."""

    def __init__(self,
                 vcpus: float,
                 memory_gib: float,
                 accelerator_count: Optional[int] = None,
                 accelerator_type: Optional[str] = None,
                 disk_gib: Optional[float] = None):
        if (accelerator_count is None) != (accelerator_type is None):
            raise ValueError('Daytona accelerator type and count must be set '
                             'together.')
        canonical = None
        if accelerator_type is not None:
            canonical = _canonical_accelerator(accelerator_type)
            if canonical is None:
                raise ValueError(
                    f'Daytona does not sell {accelerator_type!r}; it sells '
                    f'{"/".join(_GPU_PRICE_PER_HOUR)}.')
            if accelerator_count not in _GPU_COUNTS[canonical]:
                raise ValueError(f'Unsupported Daytona accelerator count '
                                 f'{canonical}:{accelerator_count}.')
        gpu_count = accelerator_count or 0
        if canonical is not None:
            max_vcpus = _MAX_VCPUS_PER_GPU * gpu_count
            max_memory = _MAX_MEMORY_GIB_PER_GPU * gpu_count
            default_disk = _DEFAULT_DISK_GIB_PER_GPU * gpu_count
            max_disk = _MAX_DISK_GIB_PER_GPU * gpu_count
        else:
            max_vcpus = _MAX_CPU_ONLY_VCPUS
            max_memory = _MAX_CPU_ONLY_MEMORY_GIB
            default_disk = _DEFAULT_CPU_DISK_GIB
            max_disk = _DEFAULT_CPU_DISK_GIB
        if not 1 <= vcpus <= max_vcpus:
            raise ValueError(
                f'Daytona vCPU request {vcpus} is out of bounds (1..'
                f'{max_vcpus} for this shape).')
        if not 1 <= memory_gib <= max_memory:
            raise ValueError(
                f'Daytona memory request {memory_gib} GiB is out of bounds '
                f'(1..{max_memory} for this shape).')
        disk_gib = default_disk if disk_gib is None else disk_gib
        if not 1 <= disk_gib <= max_disk:
            raise ValueError(
                f'Daytona disk request {disk_gib} GiB is out of bounds (1..'
                f'{max_disk} for this shape).')
        self.vcpus = vcpus
        self.memory_gib = memory_gib
        self.disk_gib = disk_gib
        self.accelerator_count = accelerator_count
        self.accelerator_type = canonical

    @property
    def name(self) -> str:
        name = (f'{_format_resource_value(self.vcpus)}CPU--'
                f'{_format_resource_value(self.memory_gib)}GB')
        if self.accelerator_type is not None:
            name += f'--{self.accelerator_type}:{self.accelerator_count}'
        return name

    @property
    def price(self) -> float:
        return _price_per_hour(self.vcpus, self.memory_gib, self.disk_gib,
                               self.accelerator_type, self.accelerator_count or
                               0)

    @classmethod
    def from_instance_type(cls, name: str) -> 'DaytonaInstanceType':
        match = _VIRTUAL_INSTANCE_TYPE_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f'Invalid Daytona instance type {name!r}.')
        count = match.group('count')
        return cls(vcpus=float(match.group('vcpus')),
                   memory_gib=float(match.group('memory')),
                   accelerator_count=int(count) if count is not None else None,
                   accelerator_type=match.group('accelerator'))


def _gpu_info(gpu_name: str, gpu_count: int) -> str:
    gpu_memory_mib = int(_GPU_MEMORY_GIB[gpu_name] * 1024)
    return repr({
        'Gpus': [{
            'Name': gpu_name,
            'Manufacturer': 'NVIDIA',
            'Count': gpu_count,
            'MemoryInfo': {
                'SizeInMiB': gpu_memory_mib,
            },
        }],
        'TotalGpuMemoryInMiB': gpu_memory_mib * gpu_count,
    })


def _make_catalog_df():
    rows = []

    def add_row(instance_type: str, region: str, price: float, vcpus: float,
                memory_gib: float, accelerator_name: Optional[str],
                accelerator_count: Optional[int],
                gpu_info: Optional[str]) -> None:
        rows.append({
            'InstanceType': instance_type,
            'AcceleratorName': accelerator_name,
            'AcceleratorCount': accelerator_count,
            'vCPUs': vcpus,
            'MemoryGiB': memory_gib,
            'Price': price,
            'Region': region,
            'GpuInfo': gpu_info,
            # Daytona really does sell preemptible GPUs, unlike Modal. The
            # multiplier is 1.0 -- an upper bound, not a discount; see above.
            'SpotPrice': (price * _SPOT_PRICE_MULTIPLIER
                          if accelerator_name is not None else None),
        })

    cpu_instance = DaytonaInstanceType(_DEFAULT_CPU_VCPUS,
                                       _DEFAULT_CPU_MEMORY_GIB)
    for region in _REGIONS:
        add_row(cpu_instance.name, region, cpu_instance.price,
                cpu_instance.vcpus, cpu_instance.memory_gib, None, None, None)

    for gpu_name, counts in _GPU_COUNTS.items():
        for gpu_count in counts:
            instance = DaytonaInstanceType(
                _DEFAULT_VCPUS_PER_GPU * gpu_count,
                _DEFAULT_MEMORY_GIB_PER_GPU * gpu_count,
                accelerator_count=gpu_count,
                accelerator_type=gpu_name)
            for region in _REGIONS:
                add_row(instance.name, region, instance.price, instance.vcpus,
                        instance.memory_gib, gpu_name, gpu_count,
                        _gpu_info(gpu_name, gpu_count))

    return pd.DataFrame(rows)


_df = _make_catalog_df()


def instance_type_exists(instance_type: str) -> bool:
    if common.instance_type_exists_impl(_df, instance_type):
        return True
    try:
        DaytonaInstanceType.from_instance_type(instance_type)
    except ValueError:
        return False
    return True


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Daytona does not support zones.')
    return common.validate_region_zone_impl(_CLOUD_NAME, _df, region, zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Daytona does not support zones.')
    if not common.instance_type_exists_impl(_df, instance_type):
        instance = DaytonaInstanceType.from_instance_type(instance_type)
        if use_spot and instance.accelerator_type is None:
            with ux_utils.print_exception_no_traceback():
                raise ValueError(
                    'Daytona spot applies to GPU sandboxes only; `spot: true` '
                    'is rejected when the sandbox requests no GPUs.')
        price = instance.price
        return price * _SPOT_PRICE_MULTIPLIER if use_spot else price
    return common.get_hourly_cost_impl(_df, instance_type, use_spot, region,
                                       zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    if not common.instance_type_exists_impl(_df, instance_type):
        instance = DaytonaInstanceType.from_instance_type(instance_type)
        return instance.vcpus, instance.memory_gib
    return common.get_vcpus_mem_from_instance_type_impl(_df, instance_type)


def _resources_from_requests(
        cpus: Optional[str],
        memory: Optional[str],
        gpu_name: Optional[str] = None,
        gpu_count: int = 0) -> DaytonaInstanceType:
    """Turn SkyPilot's ``cpus``/``memory`` requests into a Daytona shape."""
    if gpu_name is not None:
        default_vcpus = float(_DEFAULT_VCPUS_PER_GPU * gpu_count)
        default_memory = float(_DEFAULT_MEMORY_GIB_PER_GPU * gpu_count)
    else:
        default_vcpus = float(_DEFAULT_CPU_VCPUS)
        default_memory = float(_DEFAULT_CPU_MEMORY_GIB)
    vcpus = float(cpus.rstrip('+')) if cpus is not None else default_vcpus
    if memory is None:
        memory_gib = default_memory if cpus is None else vcpus * (
            default_memory / max(default_vcpus, 1))
    elif memory.endswith('+'):
        memory_gib = float(memory[:-1])
    elif memory.endswith('x'):
        memory_gib = float(memory[:-1]) * vcpus
    else:
        memory_gib = float(memory)
    # Daytona takes whole units for cpu/memory/disk; round UP so a `4+` ask is
    # never silently under-served.
    return DaytonaInstanceType(float(math.ceil(vcpus)),
                               float(math.ceil(memory_gib)),
                               accelerator_count=gpu_count or None,
                               accelerator_type=gpu_name)


def get_default_instance_type(
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None) -> Optional[str]:
    del disk_tier, local_disk  # unused
    if zone is not None:
        return None
    if use_spot:
        # Spot is GPU-only on Daytona and this path has no accelerator.
        return None
    if region is not None and region not in _REGIONS:
        return None
    try:
        instance = _resources_from_requests(cpus, memory)
    except ValueError:
        return None
    if (max_hourly_cost is not None and
            get_hourly_cost(instance.name, region=region) > max_hourly_cost):
        return None
    return instance.name


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    if not common.instance_type_exists_impl(_df, instance_type):
        instance = DaytonaInstanceType.from_instance_type(instance_type)
        if instance.accelerator_type is None:
            return None
        assert instance.accelerator_count is not None
        return {instance.accelerator_type: instance.accelerator_count}
    return common.get_accelerators_from_instance_type_impl(_df, instance_type)


def get_arch_from_instance_type(instance_type: str) -> Optional[str]:
    if not common.instance_type_exists_impl(_df, instance_type):
        DaytonaInstanceType.from_instance_type(instance_type)
        return None
    return common.get_arch_from_instance_type_impl(_df, instance_type)


def get_local_disk_from_instance_type(instance_type: str) -> Optional[str]:
    if not common.instance_type_exists_impl(_df, instance_type):
        DaytonaInstanceType.from_instance_type(instance_type)
        return None
    return common.get_local_disk_from_instance_type_impl(_df, instance_type)


def get_instance_type_for_accelerator(
    acc_name: str,
    acc_count: int,
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    use_spot: bool = False,
    local_disk: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    max_hourly_cost: Optional[float] = None
) -> Tuple[Optional[List[str]], List[str]]:
    del local_disk, use_spot  # spot is a purchase mode, not a different SKU
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Daytona does not support zones.')
    if region is not None and region not in _REGIONS:
        return None, []
    canonical = _canonical_accelerator(acc_name)
    if canonical is None or acc_count not in _GPU_COUNTS.get(canonical, ()):
        # Not a Daytona GPU. Fall through to the frame so the caller gets the
        # normal "did you mean" fuzzy list rather than a bare None.
        return common.get_instance_type_for_accelerator_impl(
            df=_df,
            acc_name=acc_name,
            acc_count=acc_count,
            cpus=None,
            memory=None,
            use_spot=False,
            region=region,
            zone=None,
            max_hourly_cost=max_hourly_cost)
    try:
        instance = _resources_from_requests(cpus,
                                            memory,
                                            gpu_name=canonical,
                                            gpu_count=acc_count)
    except ValueError:
        return [], []
    if (max_hourly_cost is not None and
            get_hourly_cost(instance.name, region=region) > max_hourly_cost):
        return [], []
    return [instance.name], []


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List[cloud.Region]:
    if not common.instance_type_exists_impl(_df, instance_type):
        DaytonaInstanceType.from_instance_type(instance_type)
        return regions()
    df = _df[_df['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


def _get_accelerator(
    accelerator: str,
    count: int,
    region: Optional[str],
    zone: Optional[str] = None,
):
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Daytona does not support zones.')
    idx = (_df['AcceleratorName'].str.fullmatch(
        accelerator, case=False)) & (_df['AcceleratorCount'] == count)
    if region is not None:
        idx &= _df['Region'] == region
    return _df[idx]


def get_accelerator_hourly_cost(accelerator: str,
                                count: int,
                                use_spot: bool = False,
                                region: Optional[str] = None,
                                zone: Optional[str] = None) -> float:
    df = _get_accelerator(accelerator, count, region, zone)
    if df.empty:
        with ux_utils.print_exception_no_traceback():
            raise ValueError(f'No accelerator {accelerator}:{count} found.')
    del use_spot  # the instance price already carries the accelerator
    return 0.0


def get_region_zones_for_accelerators(
        accelerator: str,
        count: int,
        use_spot: bool = False) -> List[cloud.Region]:
    df = _get_accelerator(accelerator, count, region=None)
    return common.get_region_zones(df, use_spot)


def check_accelerator_attachable_to_host(instance_type: str,
                                         accelerators: Optional[Dict[str, int]],
                                         zone: Optional[str] = None) -> None:
    del instance_type, accelerators  # unused
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Daytona does not support zones.')


def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    del require_price  # unused
    return common.list_accelerators_impl(_DISPLAY_NAME, _df, gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)


def regions() -> List[cloud.Region]:
    return common.get_region_zones(_df, use_spot=False)


def get_daytona_args_from_instance_type(
        instance_type: str) -> Tuple[Optional[str], int, int, int, int]:
    """Return the sandbox create args: ``(gpu_type, gpu, cpu, memory, disk)``.

    These map 1:1 onto ``POST /api/sandbox`` fields. Memory and disk are whole
    GB, cpu is whole cores -- Daytona rejects fractions.
    """
    if not instance_type_exists(instance_type):
        with ux_utils.print_exception_no_traceback():
            raise ValueError(f'No instance type {instance_type} found.')
    instance = DaytonaInstanceType.from_instance_type(instance_type)
    return (instance.accelerator_type, instance.accelerator_count or 0,
            int(instance.vcpus), int(instance.memory_gib),
            int(instance.disk_gib))
