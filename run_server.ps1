param(
    [int]$Port = 8080,
    [int]$Ctx = 32768,
    [int]$GpuLayers = 99,
    [switch]$NoMtp
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$ModelDir = 'C:\Users\niko0\.lmstudio\models\lmstudio-community\gemma-4-12B-it-QAT-GGUF'
$Model = Join-Path $ModelDir 'gemma-4-12B-it-QAT-Q4_0.gguf'
$Mmproj = Join-Path $ModelDir 'mmproj-gemma-4-12B-it-QAT-BF16.gguf'
$Mtp = Join-Path $ModelDir 'gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf'
$Media = Join-Path $PSScriptRoot 'media'
$Server = Join-Path $Root 'llama-server.exe'

if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Mmproj)) { throw "Mmproj not found: $Mmproj" }

$args = @(
    '-m', $Model,
    '-mm', $Mmproj,
    '--media-path', $Media,
    '-c', $Ctx,
    '-ngl', $GpuLayers,
    '--load-mode', 'mmap',
    '-fa', 'auto',
    '--reasoning-effort', 'none',
    '--port', $Port,
    '--host', '127.0.0.1'
)

if (-not $NoMtp) {
    if (Test-Path -LiteralPath $Mtp) {
        $args += @('--spec-type', 'draft-mtp', '--spec-draft-model', $Mtp, '--threads-draft', '8')
        Write-Host 'Speculative decoding (draft-mtp) ENABLED.'
    } else {
        Write-Host 'MTP draft model not found - running without speculative decoding.'
    }
}

Write-Host "Starting llama-server on 127.0.0.1:$Port (ctx=$Ctx, ngl=$GpuLayers)..."
& $Server @args
