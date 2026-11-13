Set-Location C:\Users\Admin\Documents\phase2_bridge
$py = 'C:\Users\Admin\carla_venv\Scripts\python.exe'
$LOG = 'v12\driver.log'
$script:n = 0
function Log($m) { ("{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) | Add-Content $LOG }
function Test-Carla { $null = & $py -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(20);c.get_world().get_map().name" 2>$null; return ($LASTEXITCODE -eq 0) }
function Start-Carla($why) {
  Log "restart CARLA: $why"
  Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep 5
  Start-Process E:\CarlaUE4\CarlaUE4.exe -ArgumentList '-windowed','-ResX=1280','-ResY=720'
  $t = 0
  while (-not (Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue) -and $t -lt 120) { Start-Sleep 5; $t++ }
  Start-Sleep 30
  $script:n = 0
}
function Test-Det($port) { $o = & $py -c "import vision_bridge as v;c=v.DetClient('127.0.0.1',$port,10);print(c.ping())" 2>$null; return (("$o") -match 'True') }
function Start-Det($why) {
  Log "start detectors: $why"
  wsl pkill -f det_server.py 2>$null
  Start-Sleep 3
  Start-Process wsl -ArgumentList 'bash','/mnt/c/Users/Admin/Documents/phase2_bridge/run_det.sh' -WindowStyle Hidden
  $t = 0
  while (-not ((Test-Det 5601) -and (Test-Det 5602)) -and $t -lt 60) { Start-Sleep 5; $t++ }
}
function HybArgs($port, $tag) { return @('--hybrid-port',"$port",'--det-conf','0.5','--det-log',"v12\detlog_$tag.jsonl") }
function Have($out) {
  $h = @{}
  if (Test-Path $out) { Get-Content $out | ForEach-Object { if ($_.Trim()) { $j = $_ | ConvertFrom-Json; $h["$($j.policy)/$($j.seed)"] = 1 } } }
  return $h
}
function Run-One($pol, $s, $w, $extra, $out, $tag, $det = 0) {
  for ($a = 1; $a -le 2; $a++) {
    if (-not (Test-Carla)) { Start-Carla "server down before $tag" } elseif ($script:n -ge 10) { Start-Carla "preventive before $tag" }
    if ($det -and -not (Test-Det $det)) { Start-Det "detector down before $tag" }
    Remove-Item tmp_run.jsonl -ErrorAction SilentlyContinue
    $ar = @('-u','carla_runner.py','--policies',$pol,'--seeds',"$s",'--inflow','0.25','--n-traffic','20','--all-red','2')
    if ($w) { $ar += @('--weather',$w) }
    $ar += $extra
    & $py @ar --out tmp_run.jsonl > "v12\log_$tag.log" 2>&1
    $script:n++
    if (Test-Path tmp_run.jsonl) {
      $l = Get-Content tmp_run.jsonl | Select-Object -Last 1
      $ok = $true
      try { $null = $l | ConvertFrom-Json } catch { $ok = $false }
      if ($ok -and $l) { Add-Content $out $l; return }
    }
    Log "FAILED attempt $a $tag"
    Start-Carla "retry $tag"
  }
  Log "GIVEUP $tag"
}

Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'carla_runner|carla_capture' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Log "start v12"
& $py patch_runner_hybrid.py *> v12\patch.txt
if ($LASTEXITCODE -ne 0) { "runner patch failed: see v12\patch.txt" | Set-Content FAILED_v12.txt; exit 1 }
Log (Get-Content v12\patch.txt -Raw)
Start-Det "initial"
if (-not ((Test-Det 5601) -and (Test-Det 5602))) { "detector servers not reachable from Windows: see v12\det_ft.log, v12\det_p1.log" | Set-Content FAILED_v12.txt; exit 1 }
Start-Carla "initial"

Remove-Item v12\gate_cam.jsonl,v12\gate_hyb.jsonl -ErrorAction SilentlyContinue
Run-One 'fixed' 0 'ClearNoon' @() 'v12\gate_cam.jsonl' 'gate_cam_nocam'
Run-One 'fixed' 0 'ClearNoon' (HybArgs 5601 'gate_cam') 'v12\gate_cam.jsonl' 'gate_cam_cam' 5601
foreach ($pol in 'preempt','fuzzy') { foreach ($i in 1..3) { Run-One $pol 0 'ClearNoon' (HybArgs 5601 "gate_${pol}_$i") 'v12\gate_hyb.jsonl' "gate_${pol}_$i" 5601 } }
& $py gate_check12.py *> v12\gate_check.txt
if ($LASTEXITCODE -ne 0) { "hybrid determinism gate failed, sweeps not run; see v12\gate_check.txt" | Set-Content FAILED_v12.txt; Log "GATE FAILED"; exit 1 }
Log "gate passed"

foreach ($c in @(@{tag='ft'; port=5601; seeds=0..9}, @{tag='p1'; port=5602; seeds=0..4})) {
  foreach ($w in 'ClearNight','ClearNoon','Fog','HardRainNoon') {
    $out = "v12\results_hyb_$($c.tag)_$w.jsonl"
    $have = Have $out
    foreach ($s in $c.seeds) { foreach ($pol in 'preempt','fuzzy') {
      if ($have.ContainsKey("$pol/$s")) { continue }
      $tg = "$($c.tag)_${w}_${pol}_$s"
      Run-One $pol $s $w (HybArgs $c.port $tg) $out "hyb_$tg" $c.port
    } }
  }
}
Log "sweeps done"

& $py analyze_v12.py v12 v11 *> v12\analysis_stdout.txt
Log "analysis done"
$bk = 'results_phase2_carla_2026-11-10'
$null = New-Item -ItemType Directory -Force $bk
Copy-Item v12 "$bk\v12" -Recurse -Force
Copy-Item det_server.py,vision_bridge.py,patch_runner_hybrid.py,analyze_v12.py,gate_check12.py,run_det.sh,run_v12.ps1,carla_runner.py,carla_runner.py.bak_pre_hybrid $bk -Force -ErrorAction SilentlyContinue
@('', '## 2026-11-10 (v12)', 'Hybrid perception added: API queue/dist/speed + detector emergency class on a camera frame (WSL server, CPU). Oracle: association, distance, speed. Protocol in v12\PROTOCOL.txt (written before results). Results v12\results_hyb_*.jsonl, analysis v12\analysis_v12.txt, gates v12\gate_check.txt.') | Add-Content STATUS.md -Encoding UTF8
wsl pkill -f det_server.py 2>$null
"done" | Set-Content done_v12.txt
Log "ALL DONE"
