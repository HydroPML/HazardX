"""Native data loading and preprocessing behind a common dataset interface."""

from __future__ import annotations
import importlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader, default_collate


def _floodcast_dataset(experiment, split, domain="australia"):
    base = experiment.base
    centered = importlib.import_module("floodcast_transolver_centered_dem_common")
    centered._configure_store(experiment.args.dem_root)
    specs = base.make_domain_specs(experiment.args)
    suffix = {"train": "", "val": "_valid", "test": "_test"}[split]
    return base.FloodDomainDataset(
        specs[domain + suffix if not domain.startswith("mozambique") else domain],
        experiment.normalizer,
        noise_std=0.0,
    )


def load_sample(experiment, split="test", index=0):
    """Read a genuine small-batch sample at the original model input resolution."""
    backend = experiment.backend
    if backend.startswith("floodcast"):
        dataset = _floodcast_dataset(experiment, split)
        sample = dataset[index]
        # The released training protocol uses complete 128x128 native tiles.
        for key in ["x", "y", "static", "mask"]:
            sample[key] = sample[key][..., :128, :128]
        sample["height"] = 128
        sample["width"] = 128
        return experiment.base.homogeneous_collate([sample])
    if backend.startswith("floodsim"):
        data = importlib.import_module("data.floodsimbench_crop1024_context6_dataset")
        common = importlib.import_module("data.floodsimbench_markov_dataset")
        city = {
            "train": common.TRAIN_CITIES[0],
            "val": common.VALID_CITIES[0],
            "test": common.TEST_CITIES[0],
        }[split]
        scenarios, _ = common.discover_scenarios(experiment.args.data_root, (city,))
        scenario = scenarios[0]
        terrain_meta = experiment.normalizer.provenance["terrain"]
        stats = terrain_meta["region_stats"][scenario.domain]
        geometry = data.SpatialGeometry(scenario.domain, *stats["shape"])
        dem = data._window_with_optional_padding(
            scenario.dem_path, geometry, 0, 0, clean_water=False
        )
        terrain = (dem - stats["terrain_mean_m"]) / terrain_meta[
            "terrain_global_train_scale"
        ]
        valid = torch.isfinite(terrain)
        terrain = torch.where(valid, terrain, torch.zeros_like(terrain))[None, None]
        depths = []
        for i in range(6):
            raw = data._window_with_optional_padding(
                scenario.frame_paths[i], geometry, 0, 0, clean_water=True
            )
            valid &= torch.isfinite(raw)
            norm = experiment.normalizer.normalize(raw, 0)
            depths.append(torch.where(valid, norm, torch.zeros_like(norm))[None, None])
        rains = [
            data.rainfall_at_frame_mm_h(scenario.rainfall_mm, i)
            for i in data.forecasting_frame_indices(0)[0]
        ]
        x = experiment.base.build_context_input(
            rains, terrain, depths[:5], valid[None, None], experiment.normalizer
        )
        return dict(
            x=x, y=depths[5], mask=valid[None, None], scenario_id=[scenario.scenario_id]
        )
    if backend == "landslide":
        cls = importlib.import_module(
            "data.landslidecast_2d_dataset"
        ).LandslideCast2DDataset
        ds = cls(experiment.args.data_path, "val" if split == "val" else split)
        sample = ds[index]
        ds.close()
        if experiment.config["model"] != "GPhyT-S-PT-FT-Euler":
            sample.pop("u0", None)
        return default_collate([sample])
    cls = importlib.import_module(
        "data.wildfire_spread_ts_dataset"
    ).WildfireSpreadTS5DayDataset
    years = {"train": [2018, 2019], "val": [2020], "test": [2021]}[split]
    args = experiment.args
    ds = cls(
        data_root=args.data_dir,
        years=years,
        stats_years=[2018, 2019],
        official_root=args.wildfire_official_root,
        split=split,
        seed=args.seed,
        backend=getattr(args, "data_backend", "tiff"),
        hdf5_root=getattr(args, "hdf5_dir", None),
    )
    return default_collate([ds[index % len(ds)]])


def evaluation_loader(experiment, split="test", protocol="source"):
    if experiment.backend == "landslide":
        return experiment.native.loader(
            experiment.args.data_path, split, experiment.args, 0, 1, False
        )[1]
    if experiment.backend == "wildfire":
        return DataLoader(
            experiment.native.datasets(experiment.args)[-1 if split == "test" else -2],
            batch_size=1 if split == "test" else experiment.args.batch_size,
            shuffle=False,
            num_workers=experiment.args.num_workers,
        )
    if experiment.backend.startswith("floodsim"):
        data = importlib.import_module(
            "data.floodsimbench_crop1024_context6_domain_balanced_dataset"
        )
        data.configure_terrain_store(experiment.args.data_root)
        cities = data.TEST_CITIES if split == "test" else data.VALID_CITIES
        return {
            city: data.FloodSimBenchContext6Dataset(
                experiment.args.data_root,
                (city,),
                "test" if split == "test" else "valid",
                experiment.normalizer,
                seed=experiment.args.seed,
            )
            for city in cities
        }
    if protocol == "mozambique":
        helper = importlib.import_module("gphyt.data.floodcast_mozambique_ood")
        specs = experiment.base.make_domain_specs(experiment.args)
        return helper.build_mozambique_protocol_datasets(
            experiment.args, specs, experiment.normalizer
        )
    return {
        domain: _floodcast_dataset(experiment, split, domain)
        for domain in ["australia", "pakistan"]
    }
