"""Daytona cloud + provisioner unit tests.

Offline: every Daytona HTTP call is stubbed. The live provisioning proof lives
in the PR description, not here -- nothing in this file spends money.
"""
import json
from unittest import mock

import pytest

from sky import clouds
from sky.catalog import daytona_catalog
from sky.provision.daytona import daytona_utils
from sky.provision.daytona import instance as daytona_instance
from sky.utils import command_runner
from sky.utils import registry
from sky.utils import resources_utils


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(daytona_utils.API_KEY_ENV_VAR, 'dtn_test')
    monkeypatch.delenv(daytona_utils.ORG_ID_ENV_VAR, raising=False)


class TestRegistration:

    def test_cloud_is_registered(self):
        assert registry.CLOUD_REGISTRY.from_str('daytona') is not None
        assert isinstance(clouds.Daytona(), clouds.Cloud)

    def test_in_all_clouds_so_infra_daytona_validates(self):
        # The `infra:` schema pattern is generated from ALL_CLOUDS; without
        # this entry `infra: daytona` fails YAML validation before it ever
        # reaches the optimizer.
        from sky.skylet import constants
        from sky.utils import schemas
        assert 'daytona' in constants.ALL_CLOUDS
        assert 'daytona' in schemas._get_infra_pattern()

    def test_has_a_ray_template(self):
        from sky.backends import cloud_vm_ray_backend
        mapping = cloud_vm_ray_backend._get_cluster_config_template
        assert mapping(clouds.Daytona()) == 'daytona-ray.yml.j2'


class TestCatalog:

    def test_h100_resolves_with_a_sane_shape(self):
        got, _ = daytona_catalog.get_instance_type_for_accelerator('H100', 1)
        assert got == ['4CPU--16GB--H100:1']
        gpu_type, gpu, cpu, mem, disk = (
            daytona_catalog.get_daytona_args_from_instance_type(got[0]))
        assert (gpu_type, gpu, cpu, mem, disk) == ('H100', 1, 4, 16, 50)

    def test_price_is_all_in_not_gpu_only(self):
        # Daytona bills vCPU/RAM/disk on top of the GPU, and its RAM rate is
        # high enough to dominate: pricing on the GPU rate alone understates a
        # node badly. 2.27 is the bare H100 rate.
        price = daytona_catalog.get_hourly_cost('4CPU--16GB--H100:1')
        assert price > 2.27
        assert price == pytest.approx(2.27 + 4 * 0.0504 + 16 * 0.0162 +
                                      50 * 0.000108)

    def test_spot_is_offered_and_never_cheaper_than_on_demand(self):
        # Daytona really does sell preemptible GPUs (unlike Modal), but the
        # published spot column is not machine-readable, so the catalog quotes
        # the on-demand rate as an upper bound rather than inventing a
        # discount that would win optimizer decisions it has not earned.
        od = daytona_catalog.get_hourly_cost('4CPU--16GB--H100:1')
        spot = daytona_catalog.get_hourly_cost('4CPU--16GB--H100:1',
                                               use_spot=True)
        assert spot >= od

    def test_scales_with_gpu_count(self):
        got, _ = daytona_catalog.get_instance_type_for_accelerator('H100', 2)
        assert got == ['8CPU--32GB--H100:2']
        one = daytona_catalog.get_hourly_cost('4CPU--16GB--H100:1')
        two = daytona_catalog.get_hourly_cost(got[0])
        assert two == pytest.approx(2 * one)

    def test_a100_is_not_sold(self):
        # A100 is in common accelerator chains and Daytona has never sold one.
        # It must answer "no", not substitute something else.
        got, _ = daytona_catalog.get_instance_type_for_accelerator('A100', 1)
        assert not got

    def test_gpu_ceilings_match_the_live_api(self):
        # Confirmed against GET /organizations/{id}/usage, which reports
        # maxCpuPerGpu=16, maxMemoryPerGpu=192, maxDiskPerGpu=512.
        assert daytona_catalog._MAX_VCPUS_PER_GPU == 16
        assert daytona_catalog._MAX_MEMORY_GIB_PER_GPU == 192
        assert daytona_catalog._MAX_DISK_GIB_PER_GPU == 512

    def test_over_ceiling_is_refused(self):
        with pytest.raises(ValueError):
            daytona_catalog.DaytonaInstanceType(64, 16, 1, 'H100')


class TestCloudFeatures:

    def test_spot_is_supported_for_gpus(self):
        r = mock.Mock()
        r.accelerators = {'H100': 1}
        unsupported = clouds.Daytona()._unsupported_features_for_resources(r)
        assert (clouds.CloudImplementationFeatures.SPOT_INSTANCE
                not in unsupported)

    def test_spot_is_unsupported_without_a_gpu(self):
        # Daytona rejects `spot: true` on a sandbox that requests no GPUs.
        r = mock.Mock()
        r.accelerators = None
        unsupported = clouds.Daytona()._unsupported_features_for_resources(r)
        assert (clouds.CloudImplementationFeatures.SPOT_INSTANCE
                in unsupported)

    def test_stop_is_unsupported_because_stopping_deletes(self):
        r = mock.Mock()
        r.accelerators = {'H100': 1}
        unsupported = clouds.Daytona()._unsupported_features_for_resources(r)
        assert clouds.CloudImplementationFeatures.STOP in unsupported
        assert clouds.CloudImplementationFeatures.MULTI_NODE in unsupported

    def test_credentials_never_travel_to_the_node(self):
        # A sandbox that can create sandboxes can spend the whole account.
        assert clouds.Daytona().get_credential_file_mounts() == {}


class TestCreateBody:

    def _body(self, node_config, monkeypatch):
        seen = {}

        def fake(method, path, params=None, body=None, **kw):
            seen['method'], seen['path'], seen['body'] = method, path, body
            return {'id': 'sbx-1'}

        monkeypatch.setattr(daytona_utils, '_request', fake)
        daytona_utils.create_sandbox('cluster-1', node_config)
        return seen['body']

    def _gpu_config(self, **over):
        cfg = {
            'Cpu': 4,
            'Memory': 16,
            'Disk': 50,
            'Gpu': 1,
            'GpuType': 'H100',
            'UseSpot': True,
            'TtlMinutes': 60,
            'DefaultImage': 'pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime',
        }
        cfg.update(over)
        return cfg

    def test_gpu_sandbox_is_ephemeral_and_deadlined(self, api_key, monkeypatch):
        body = self._body(self._gpu_config(), monkeypatch)
        # A GPU sandbox MUST be autoDeleteInterval=0; Daytona rejects anything
        # else. autoStopInterval=0 stops a quiet training poll being reaped.
        assert body['autoDeleteInterval'] == 0
        assert body['autoStopInterval'] == 0
        # The teardown that survives the API server dying.
        assert body['ttlMinutes'] == 60
        assert body['spot'] is True
        assert body['gpuType'] == ['H100']

    def test_image_goes_as_a_build_context_not_an_image_field(
            self, api_key, monkeypatch):
        # POST /sandbox has NO `image` field: sending one is silently ignored,
        # the platform falls back to its default snapshot, and the request is
        # then rejected for specifying resources alongside a snapshot.
        body = self._body(self._gpu_config(), monkeypatch)
        assert 'image' not in body
        docker = body['buildInfo']['dockerfileContent']
        assert docker.startswith(
            'FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime')
        # curl and patch are installed at BUILD time, where egress is
        # unrestricted; inside the sandbox it is not.
        assert 'curl' in docker and 'patch' in docker

    def test_snapshot_path_drops_resources(self, api_key, monkeypatch):
        body = self._body(self._gpu_config(Snapshot='daytona-gpu'),
                          monkeypatch)
        assert body['snapshot'] == 'daytona-gpu'
        for key in ('cpu', 'memory', 'disk', 'gpu', 'gpuType'):
            assert key not in body

    def test_labelled_so_teardown_finds_only_its_own(self, api_key,
                                                     monkeypatch):
        body = self._body(self._gpu_config(), monkeypatch)
        assert body['labels'][daytona_utils.CLUSTER_LABEL] == 'cluster-1'
        assert body['labels'][daytona_utils.OWNER_LABEL] == 'skypilot'

    def test_spot_is_not_sent_for_a_cpu_sandbox(self, api_key, monkeypatch):
        body = self._body(self._gpu_config(Gpu=0, GpuType=None), monkeypatch)
        assert 'spot' not in body
        assert 'gpu' not in body

    def test_domain_allow_list_is_capped(self, api_key, monkeypatch):
        monkeypatch.setattr(daytona_utils, '_request',
                            lambda *a, **k: {'id': 'x'})
        cfg = self._gpu_config(
            DomainAllowList=[f'd{i}.example.com' for i in range(25)])
        with pytest.raises(daytona_utils.DaytonaError, match='at most 20'):
            daytona_utils.create_sandbox('c', cfg)

    def test_default_allow_list_fits(self):
        assert (len(daytona_utils.DEFAULT_DOMAIN_ALLOW_LIST) <=
                daytona_utils.MAX_DOMAIN_ALLOW_LIST)


class TestCommandRunner:

    def test_node_id_is_the_sandbox(self):
        r = command_runner.DaytonaCommandRunner(('sbx-1', 'us'))
        # One sandbox is one container: H100:2 is two GPUs on ONE node. The
        # evsys pools model derives the same id.
        assert r.node_id == 'daytona-sbx-1'

    def test_no_argv_cap_like_modal(self):
        # Measured live: a 1,000,000-byte command ran correctly, and 1 MB of
        # stdout came back whole. Modal caps argv near 8 KB and fails silently.
        r = command_runner.DaytonaCommandRunner(('sbx-1', 'us'))
        assert r.max_inline_command_length() >= 512 * 1024

    def test_port_forward_is_refused_with_the_alternative(self):
        r = command_runner.DaytonaCommandRunner(('sbx-1', 'us'))
        with pytest.raises(NotImplementedError, match='preview'):
            r.port_forward_command([(1, 2)])

    def test_file_upload_targets_the_destination_path(self, api_key,
                                                      monkeypatch, tmp_path):
        # The bug class this avoids: deriving the destination from the
        # SOURCE's basename lands a file under its local temp name instead of
        # at its target (ev-sys/skypilot#4).
        src = tmp_path / 'tmpXYZ123'
        src.write_text('payload')
        seen = {}

        def fake(method, path, params=None, files=None, **kw):
            seen['path'] = path
            seen['params'] = params
            return {}

        monkeypatch.setattr(daytona_utils, '_toolbox_request', fake)
        daytona_utils.upload_file('sbx-1', str(src), '/root/.sky/target.yaml')
        assert seen['params']['path'] == '/root/.sky/target.yaml'


class TestProvisioner:

    def test_multi_node_is_refused_up_front(self, api_key):
        cfg = mock.Mock()
        cfg.count = 2
        with pytest.raises(RuntimeError, match='single-node'):
            daytona_instance.run_instances('us', 'c', 'c-1', cfg)

    def test_stop_says_why_it_cannot(self, api_key):
        with pytest.raises(NotImplementedError, match='DELETED when it stops'):
            daytona_instance.stop_instances('c-1')

    def test_quota_preflight_is_free_and_refuses_early(self, api_key,
                                                       monkeypatch):
        monkeypatch.setattr(
            daytona_utils, 'organization_usage', lambda: [{
                'regionId': 'us',
                'sandboxClass': 'container',
                'totalGpuQuota': 2,
                'currentGpuUsage': 2,
            }])
        problem = daytona_utils.check_gpu_quota('us', 1)
        assert problem is not None and 'quota' in problem

    def test_unreadable_usage_is_unknown_not_no_room(self, api_key,
                                                    monkeypatch):
        # A write:sandboxes-only key gets 403 on usage. That must not read as
        # "the org is full".
        monkeypatch.setattr(daytona_utils, 'organization_usage', lambda: [])
        assert daytona_utils.check_gpu_quota('us', 8) is None

    def test_cluster_info_has_no_ssh_port(self, api_key, monkeypatch):
        # HTTPS-only by design; ssh_port=None is what makes provisioner.py
        # skip its SSH probe for this cloud.
        monkeypatch.setattr(daytona_utils, 'find_sandboxes',
                            lambda *a, **k: [{'id': 'sbx-1', 'spot': True,
                                              'createdAt': '2026-01-01'}])
        info = daytona_instance.get_cluster_info('us', 'c-1')
        head = info.get_head_instance()
        assert head.ssh_port is None
        assert head.external_ip is None
        assert info.instances['sbx-1'][0].internal_ip == '127.0.0.1'

    def test_command_runners_are_https(self, api_key, monkeypatch):
        monkeypatch.setattr(daytona_utils, 'find_sandboxes',
                            lambda *a, **k: [{'id': 'sbx-1',
                                              'createdAt': '2026-01-01'}])
        info = daytona_instance.get_cluster_info('us', 'c-1')
        runners = daytona_instance.get_command_runners(info)
        assert len(runners) == 1
        assert isinstance(runners[0], command_runner.DaytonaCommandRunner)

    def test_preemption_is_reported_as_the_reason(self, api_key, monkeypatch):
        monkeypatch.setattr(
            daytona_utils, 'find_sandboxes', lambda *a, **k: [{
                'id': 'sbx-1',
                'state': 'destroyed',
                'spotEvictedAt': '2026-08-25T00:00:00Z',
                'createdAt': '2026-01-01',
            }])
        got = daytona_instance.query_instances('c', 'c-1',
                                               non_terminated_only=False)
        status, reason = got['sbx-1']
        assert status is None
        assert 'preemption' in reason


class TestDeployVariables:

    def test_disk_size_wins_over_the_catalog_default(self, api_key):
        d = clouds.Daytona()
        import sky
        r = sky.Resources(infra='daytona', accelerators='H100:1',
                          disk_size=200)
        feas = d._get_feasible_launchable_resources(r).resources_list[0]
        feas = feas.copy(region='us')
        variables = d.make_deploy_resources_variables(
            feas, resources_utils.ClusterName('c', 'c-1'),
            clouds.Region('us'), None, 1)
        assert variables['daytona_disk'] == 200
        assert variables['daytona_gpu_type'] == 'H100'
        assert variables['daytona_gpu'] == 1
