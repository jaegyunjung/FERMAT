# End-to-end Synthetic SNUH smoke harness for FERMAT (Windows PowerShell).
#
# Behavior:
#   1. If $env:SYNTHETIC_SNUH_DUCKDB is set and the file exists, run
#      inspect + preprocess on the real Synthetic SNUH DuckDB file.
#      Otherwise, default to data\synthetic_snuh_raw.duckdb.
#   2. If that file does not exist either, fall back to
#      scripts\generate_self_synthetic_4col.py.
#
# Always then:
#   3. Validate bin files (check_fermat_bin.py)
#   4. Compute summary statistics (summarize_fermat_dataset.py)
#   5. Run smoke training (config/train_fermat_synthetic_snuh.py)
#   6. Run 3-column Delphi regression test (config/train_fermat_demo.py)
#
# Usage:
#     # Default path (data\synthetic_snuh_raw.duckdb):
#     powershell -ExecutionPolicy Bypass -File scripts\run_smoke_synthetic_snuh.ps1
#
#     # Override path:
#     $env:SYNTHETIC_SNUH_DUCKDB = "C:\path\to\my.duckdb"
#     powershell -ExecutionPolicy Bypass -File scripts\run_smoke_synthetic_snuh.ps1
#
# Note: requires `python` to resolve to a Python 3 interpreter with
# torch, numpy<2.0, pandas, duckdb, tqdm installed.

$ErrorActionPreference = "Stop"

# Ensure log/data directories exist
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\synthetic_snuh" | Out-Null

# Resolve DuckDB path
if ($env:SYNTHETIC_SNUH_DUCKDB) {
    $DuckdbPath = $env:SYNTHETIC_SNUH_DUCKDB
} else {
    $DuckdbPath = "data\synthetic_snuh_raw.duckdb"
}

function Tee-Run {
    param([string]$LogPath, [scriptblock]$Block)
    & $Block 2>&1 | Tee-Object -FilePath $LogPath
}

if (Test-Path $DuckdbPath) {
    Write-Host "[harness] using real Synthetic SNUH at $DuckdbPath"

    Write-Host "[harness] step 1: inspect schema"
    Tee-Run -LogPath "logs\01_inspect.log" -Block {
        python scripts\inspect_synthetic_snuh_schema.py `
            --duckdb $DuckdbPath `
            --out logs\synthetic_snuh_schema_summary.txt
    }

    Write-Host "[harness] step 2: preprocess to 4-column"
    Tee-Run -LogPath "logs\02_preprocess.log" -Block {
        python scripts\preprocess_synthetic_snuh_to_fermat.py `
            --duckdb $DuckdbPath `
            --out_dir data\synthetic_snuh `
            --vocab_cap 2000 `
            --val_frac 0.1 `
            --seed 42
    }
} else {
    Write-Host "[harness] $DuckdbPath not found — falling back to self-synthetic generator"
    Tee-Run -LogPath "logs\02_preprocess.log" -Block {
        python scripts\generate_self_synthetic_4col.py `
            --out_dir data\synthetic_snuh `
            --n_patients 2000 `
            --seed 42
    }
}

Write-Host "[harness] step 3: validate bin"
Tee-Run -LogPath "logs\03_check_bin.log" -Block {
    python scripts\check_fermat_bin.py `
        --data_dir data\synthetic_snuh `
        --out logs\fermat_bin_check.log
}

Write-Host "[harness] step 4: summarize dataset"
Tee-Run -LogPath "logs\04_summary.log" -Block {
    python scripts\summarize_fermat_dataset.py `
        --data_dir data\synthetic_snuh `
        --out_txt logs\fermat_dataset_summary.txt `
        --out_md logs\fermat_dataset_summary.md
}

Write-Host "[harness] step 5: smoke training (4-column)"
Tee-Run -LogPath "logs\fermat_synthetic_snuh_smoke_test.log" -Block {
    python train.py config\train_fermat_synthetic_snuh.py --device=cpu
}

Write-Host "[harness] step 6: 3-column Delphi regression"
Tee-Run -LogPath "logs\fermat_3col_regression_test.log" -Block {
    python train.py config\train_fermat_demo.py --device=cpu --max_iters=50
}

Write-Host "[harness] done. logs\ contains all artifacts."
