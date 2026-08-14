# Convenience runner (PowerShell)
param([string]$cmd = "ui")

if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

switch ($cmd) {
    "install" { pip install -r requirements.txt }
    "ingest"  { python -m ronin.ingest }
    "ui"      { streamlit run app.py }
    "api"     { uvicorn ronin.api:app --reload }
    "eval"    { python -m ronin.eval.run_eval }
    default   { Write-Host "Usage: .\run.ps1 [install|ingest|ui|api|eval]" }
}
