$Arguments = $args
$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $localPython) { $pythonExe = $localPython }
elseif (Test-Path -LiteralPath $bundledPython) { $pythonExe = $bundledPython }
else { $pythonExe = (Get-Command python -ErrorAction Stop).Source }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.packages')) {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot '.packages') + [IO.Path]::PathSeparator + $env:PYTHONPATH
}
Push-Location $PSScriptRoot
try { & $pythonExe @Arguments; $code = $LASTEXITCODE } finally { Pop-Location }
exit $code
