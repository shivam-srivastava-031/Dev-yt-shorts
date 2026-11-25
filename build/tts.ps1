# Renders narration lines to WAV with the built-in Windows voice (scratch track).
# Usage: powershell -File build/tts.ps1 -Job path\to\job.json
# job.json = [{ "id": "...", "text": "...", "path": "C:\\...\\file.wav" }, ...]
param(
    [Parameter(Mandatory = $true)][string]$Job,
    [string]$Voice = "Microsoft David Desktop",
    [int]$Rate = -1
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 $Job | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $synth.SelectVoice($Voice) } catch { Write-Warning "Voice '$Voice' not found; using the default voice." }
$synth.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($it in $items) {
    $synth.SetOutputToWaveFile($it.path, $fmt)
    $synth.Speak($it.text)
    $synth.SetOutputToNull()
}
$synth.Dispose()
