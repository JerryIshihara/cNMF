#!/usr/bin/env python3
"""Generate the deterministic synthetic three-program cNMF fixture."""

from argparse import ArgumentParser
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_SPEC = Path(__file__).resolve().parent / "run_spec.json"


def load_spec(path=DEFAULT_SPEC):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def generate_dataset(spec):
    """Return counts and their known three-program generative factors."""
    fixture = spec["fixture"]
    seed = fixture["seed"]
    n_programs = fixture["programs"]
    n_genes = fixture["genes"]
    if (n_programs, n_genes) != (3, 12):
        raise ValueError("the synthetic generator currently requires 3 programs and 12 genes")
    rng = np.random.default_rng(seed)

    spectra = np.full((n_programs, n_genes), 0.2, dtype=np.float64)
    spectra[0, 0:4] = [8.0, 6.0, 4.0, 2.0]
    spectra[1, 4:8] = [7.0, 5.0, 4.0, 2.0]
    spectra[2, 8:12] = [6.0, 5.0, 3.0, 2.0]

    group_sizes = fixture["cell_groups"]
    labels = [
        label
        for label in ("program_1", "program_2", "program_3", "mixed")
        for _ in range(group_sizes[label])
    ]
    if len(labels) != fixture["cells"]:
        raise ValueError("cell_groups must sum to fixture.cells")
    usages = np.empty((len(labels), n_programs), dtype=np.float64)
    dominant_usage = {
        "program_1": np.array([2.8, 0.15, 0.10]),
        "program_2": np.array([0.15, 2.2, 0.10]),
        "program_3": np.array([0.10, 0.15, 1.6]),
    }
    mixed_usages = np.array(
        [
            [1.4, 1.1, 0.6],
            [1.2, 0.8, 1.0],
            [0.8, 1.4, 0.7],
            [1.0, 0.7, 1.3],
            [1.5, 0.9, 0.5],
            [0.7, 1.2, 1.1],
        ]
    )

    mixed_index = 0
    for cell_index, label in enumerate(labels):
        if label == "mixed":
            usages[cell_index] = mixed_usages[mixed_index]
            mixed_index += 1
        else:
            scale = 0.85 + 0.05 * (cell_index % 7)
            usages[cell_index] = dominant_usage[label] * scale

    expected_rate = usages @ spectra + 0.5
    counts = rng.poisson(expected_rate).astype(np.int32)
    return counts, usages, spectra, labels


def write_dataset(output_dir, spec):
    """Write the synthetic AnnData input and inspectable truth tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts, usages, spectra, labels = generate_dataset(spec)
    cell_ids = [f"cell_{index:03d}" for index in range(counts.shape[0])]
    gene_ids = [f"gene_{index:02d}" for index in range(counts.shape[1])]
    program_names = [
        f"program_{index}" for index in range(1, spec["fixture"]["programs"] + 1)
    ]

    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs["true_program"] = pd.Categorical(
        labels,
        categories=program_names + ["mixed"],
    )
    obs["library_size"] = counts.sum(axis=1)

    var = pd.DataFrame(index=pd.Index(gene_ids, name="gene_id"))
    var["marker_program"] = pd.Categorical(
        [program_names[index // 4] for index in range(spec["fixture"]["genes"])],
        categories=program_names,
    )

    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.obsm["true_usage"] = usages
    adata.varm["true_spectra"] = spectra.T
    adata.uns["fixture_seed"] = spec["fixture"]["seed"]
    adata.uns["program_names"] = program_names
    adata.write_h5ad(output_dir / "synthetic_counts.h5ad")

    pd.DataFrame(counts, index=cell_ids, columns=gene_ids).to_csv(
        output_dir / "synthetic_counts.tsv",
        sep="\t",
        index_label="cell_id",
    )
    pd.DataFrame(usages, index=cell_ids, columns=program_names).to_csv(
        output_dir / "true_usages.tsv",
        sep="\t",
        index_label="cell_id",
    )
    pd.DataFrame(spectra, index=program_names, columns=gene_ids).to_csv(
        output_dir / "true_spectra.tsv",
        sep="\t",
        index_label="program",
    )
    (output_dir / "genes.txt").write_text(
        "\n".join(gene_ids) + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote {counts.shape[0]} cells x {counts.shape[1]} genes "
        f"with seed {spec['fixture']['seed']} to {output_dir}"
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    write_dataset(args.output_dir, load_spec(args.spec))


if __name__ == "__main__":
    main()
