# Instala o Margem de Segurança no Windows:
#  - ambiente Python e cache fora do OneDrive, em %LOCALAPPDATA%\margem_de_seguranca
#  - dependências com as versões exatas do desenvolvimento
#  - a skill do Claude Code no nível do usuário (%USERPROFILE%\.claude\skills\margem-de-seguranca)
# Rode na pasta do projeto:  powershell -ExecutionPolicy Bypass -File .\instalar.ps1

$ErrorActionPreference = "Stop"
$projeto = $PSScriptRoot
$base = Join-Path $env:LOCALAPPDATA "margem_de_seguranca"
$venv = Join-Path $base ".venv"
$python = Join-Path $venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $base | Out-Null
if (-not (Test-Path $python)) {
    Write-Host "Criando o ambiente Python em $venv ..."
    py -3.13 -m venv $venv 2>$null
    if (-not (Test-Path $python)) { py -3 -m venv $venv }
}
& $python -m pip install --upgrade pip | Out-Null
Write-Host "Instalando dependências..."
& $python -m pip install -r (Join-Path $projeto "requirements-lock.txt")
& $python -m pip install -e $projeto --no-deps

$skillOrigem = Join-Path $projeto "skill\margem-de-seguranca\SKILL.md"
$skillDestino = Join-Path $env:USERPROFILE ".claude\skills\margem-de-seguranca"
New-Item -ItemType Directory -Force -Path $skillDestino | Out-Null
$texto = Get-Content $skillOrigem -Raw -Encoding UTF8
$texto = $texto.Replace("{{PASTA_PROJETO}}", $projeto).Replace("{{PYTHON}}", $python)
Set-Content -Path (Join-Path $skillDestino "SKILL.md") -Value $texto -Encoding UTF8

$env_arquivo = Join-Path $base ".env"
if (-not (Test-Path $env_arquivo)) {
    Set-Content -Path $env_arquivo -Value "# Chave gratuita do FRED (fred.stlouisfed.org), para as versões da época (ALFRED):`n# FRED_API_KEY=" -Encoding UTF8
}
Write-Host ""
Write-Host "Pronto. Skill instalada em $skillDestino"
Write-Host "Teste:  & `"$python`" -m margem ranking-dos-ativos"
