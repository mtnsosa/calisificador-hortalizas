<#
.SYNOPSIS
    Genera el paquete limpio de entrega profesional clasificador_hortalizas_entrega_act4_2.zip
    (Actividad 4.2 — Empaquetado y Despliegue).

.DESCRIPTION
    Incluye ÚNICAMENTE los artefactos productivos oficiales sin tocar ni borrar
    los directorios históricos de desarrollo (venv/, data/raw/, data/splits/, modelos/, etc.).
    Utiliza rutas POSIX ('/') dentro del archivo ZIP para garantizar compatibilidad
    total en Windows, Linux, macOS y contenedores Docker.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$ZipNombre = "clasificador_hortalizas_entrega_act4_2.zip"
$ZipRuta = Join-Path $PSScriptRoot $ZipNombre

# Lista blanca estricta de archivos individuales obligatorios
$ArchivosIndividuales = @(
    "README.md",
    "requirements.txt",
    "requirements_ui.txt",
    "Dockerfile",
    ".dockerignore",
    "run_demo.bat",
    "run_demo.sh",
    "run_ui.bat",
    "run_ui.sh",
    "models/modelo_hortalizas_act3_3.keras",
    "models/model_metadata.json",
    "data/clases.txt",
    "docs/manual_instalacion.md",
    "outputs/.gitkeep"
)

# Validar existencia previa de todos los artefactos obligatorios
foreach ($rel in $ArchivosIndividuales) {
    $abs = Join-Path $PSScriptRoot $rel
    if (-not (Test-Path -LiteralPath $abs -PathType Leaf)) {
        throw "Falta el archivo obligatorio para el empaquetado: $rel"
    }
}

# Recolectar scripts productivos en src/ (excluyendo __pycache__ y *.pyc)
$ArchivosSrc = Get-ChildItem -Path (Join-Path $PSScriptRoot "src") -Filter "*.py" -File | Sort-Object Name
if ($ArchivosSrc.Count -eq 0) {
    throw "No se encontraron scripts Python en src/."
}

# Recolectar imágenes de ejemplo en data/ejemplos/
$ExtensionesImg = @(".jpg", ".jpeg", ".png")
$ArchivosEjemplos = Get-ChildItem -Path (Join-Path $PSScriptRoot "data/ejemplos") -File |
    Where-Object { $ExtensionesImg -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object Name

if ($ArchivosEjemplos.Count -lt 6) {
    throw "Se esperaban al menos 6 imágenes en data/ejemplos/, encontradas: $($ArchivosEjemplos.Count)"
}

# Construir lista consolidada de entradas (RutaAbsoluta -> RutaRelativaPOSIX)
$EntradasZip = New-Object System.Collections.Generic.List[PSCustomObject]

foreach ($rel in $ArchivosIndividuales) {
    $EntradasZip.Add([PSCustomObject]@{
        Origen  = (Join-Path $PSScriptRoot $rel)
        Destino = ($rel -replace "\\", "/")
    })
}

foreach ($f in $ArchivosSrc) {
    $EntradasZip.Add([PSCustomObject]@{
        Origen  = $f.FullName
        Destino = "src/$($f.Name)"
    })
}

foreach ($img in $ArchivosEjemplos) {
    $EntradasZip.Add([PSCustomObject]@{
        Origen  = $img.FullName
        Destino = "data/ejemplos/$($img.Name)"
    })
}

# Eliminar paquete previo si existiera
if (Test-Path -LiteralPath $ZipRuta) {
    Remove-Item -LiteralPath $ZipRuta -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$FileStream = [System.IO.File]::Open($ZipRuta, [System.IO.FileMode]::CreateNew)
try {
    $ZipArchive = New-Object System.IO.Compression.ZipArchive($FileStream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($item in $EntradasZip) {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $ZipArchive,
                $item.Origen,
                $item.Destino,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $ZipArchive.Dispose()
    }
}
finally {
    $FileStream.Dispose()
}

$TamanoMB = [Math]::Round((Get-Item -LiteralPath $ZipRuta).Length / 1MB, 2)
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " PAQUETE DE ENTREGA GENERADO EXITOSAMENTE" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Archivo : $ZipRuta"
Write-Host " Tamaño  : $TamanoMB MB"
Write-Host " Entradas: $($EntradasZip.Count) archivos empaquetados"
Write-Host "------------------------------------------------------------------"
foreach ($item in $EntradasZip) {
    Write-Host ("  [+] " + $item.Destino)
}
Write-Host "==================================================================" -ForegroundColor Cyan
