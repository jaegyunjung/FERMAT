param([Parameter(Mandatory=$true)][string]$SinceIso)
$ErrorActionPreference='Stop'
Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop | Out-Null
$taskEvents=@()
try {
    $taskEvents=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=[datetime]$SinceIso;Level=1,2,3;ProviderName=@('Microsoft-Windows-WHEA-Logger','nvlddmkm','Microsoft-Windows-WER-SystemErrorReporting','Display')} -ErrorAction Stop | Where-Object {$_.ProviderName -ne 'Display' -or $_.Id -eq 4101})
} catch {
    if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') { throw }
}
@{events=@($taskEvents | ForEach-Object {@{id=$_.Id;provider=$_.ProviderName;record_id=$_.RecordId}});
  boot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o');
  reboot_required=(New-Object -ComObject Microsoft.Update.SystemInfo).RebootRequired
} | ConvertTo-Json -Depth 4 -Compress
