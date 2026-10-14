param(
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [int]$Samples = 10
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing to overwrite $OutputPath" }
if ($Samples -lt 1) { throw 'Samples must be positive' }
$battery = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus |
    Select-Object PowerOnline, Charging, Discharging
$gpuSamples = @(for ($sampleIndex = 0; $sampleIndex -lt $Samples; $sampleIndex++) {
    $raw = & nvidia-smi --query-gpu=name,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.gr,clocks.mem --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    [ordered]@{ timestamp_utc = [DateTime]::UtcNow.ToString('o'); csv = @($raw) }
    if ($sampleIndex -lt $Samples - 1) { Start-Sleep -Seconds 1 }
})
$processes = @(& nvidia-smi --query-compute-apps=process_name --format=csv,noheader |
    ForEach-Object { [System.IO.Path]::GetFileName($_.Trim()) } | Sort-Object -Unique)
$context = [ordered]@{
    timestamp_utc = [DateTime]::UtcNow.ToString('o')
    battery = @($battery)
    active_power_scheme = (& powercfg /getactivescheme | Out-String).Trim()
    gpu_sample_csv_fields = @('name', 'pstate', 'power_w', 'temperature_c', 'utilization_percent', 'memory_mib', 'graphics_clock_mhz', 'memory_clock_mhz')
    gpu_samples = $gpuSamples
    reported_gpu_process_names = $processes
    note = 'Snapshot only; process presence does not establish GPU load. Not continuous monitoring. No power settings changed.'
}
$destination = [System.IO.Path]::GetFullPath($OutputPath)
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination)) | Out-Null
[System.IO.File]::WriteAllText($destination, ($context | ConvertTo-Json -Depth 8) + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Output $destination
