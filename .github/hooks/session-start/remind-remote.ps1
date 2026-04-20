# Session Start: /remote Reminder
# Logs session start and outputs a reminder to enable /remote.
# Output from sessionStart hooks is technically ignored by the agent,
# but this serves as a deterministic log and belt-and-suspenders approach.
$ErrorActionPreference = "SilentlyContinue"

try {
    $hookInput = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $source = $hookInput.source
    $timestamp = $hookInput.timestamp

    $logDir = Join-Path $PSScriptRoot "..\..\..\..\logs"
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

    $logEntry = @{
        timestamp = (Get-Date -Format "o")
        event     = "session-start-remote-reminder"
        source    = $source
    } | ConvertTo-Json -Compress

    $logEntry | Out-File -Append (Join-Path $logDir "hooks.jsonl") -Encoding utf8

    Write-Output "Reminder: Run /remote to enable remote control from GitHub web and mobile."
    exit 0
} catch {
    exit 0
}
